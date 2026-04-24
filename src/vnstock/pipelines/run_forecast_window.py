from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from vnstock.evaluation.compare import load_all_predictions
from vnstock.models.kronos.zero_shot import _load_kronos_classes
from vnstock.utils.io import ensure_dir, load_table, read_yaml
from vnstock.utils.logging import get_logger
from vnstock.utils.paths import path_for


DEFAULT_SYMBOLS = ["VIC", "FPT", "VHM"]
ENDPOINT_MODELS = ["xlstm_ts", "itransformer"]
MODEL_COLORS = {
    "xlstm_ts": "#2563eb",
    "itransformer": "#f97316",
    "kronos": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a forecast-origin window with 5 future sessions actual vs predictions."
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--origin-date", default="latest")
    parser.add_argument("--history-length", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--raw-panel", default=None)
    parser.add_argument("--kronos-config", default="configs/models/kronos.yaml")
    parser.add_argument("--kronos-repo", default="external/Kronos")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figures-dir", default="outputs/figures/model_compare/forecast_window")
    parser.add_argument("--reports-dir", default="outputs/reports/model_compare/forecast_window")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_forecast_window")
    figures_dir = Path(args.figures_dir)
    reports_dir = Path(args.reports_dir)
    if args.clean:
        _clean_output_dir(figures_dir)
        _clean_output_dir(reports_dir)
    figures_dir = ensure_dir(figures_dir)
    reports_dir = ensure_dir(reports_dir)

    raw_panel_path = Path(args.raw_panel) if args.raw_panel else path_for("interim_root") / "cleaned_panel.parquet"
    _set_seed(args.seed)
    raw_panel = load_table(raw_panel_path).copy()
    raw_panel["date"] = pd.to_datetime(raw_panel["date"])
    raw_panel = raw_panel.sort_values(["symbol", "date"])

    prediction_frame = load_all_predictions(args.predictions_dir)
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])
    prediction_frame = prediction_frame.loc[prediction_frame["split"].astype(str) == args.split].copy()
    prediction_frame = prediction_frame.loc[prediction_frame["model_family"].isin(ENDPOINT_MODELS)].copy()
    if prediction_frame.empty:
        raise ValueError("No endpoint model predictions found for xLSTM-TS/iTransformer.")

    origin_date = _resolve_origin_date(
        prediction_frame=prediction_frame,
        raw_panel=raw_panel,
        symbols=args.symbols,
        requested_origin=args.origin_date,
        horizon=args.horizon,
    )

    kronos_predictor, kronos_settings = _build_kronos_predictor(args)
    forecast_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, str]] = []

    for symbol in args.symbols:
        symbol_raw = raw_panel.loc[raw_panel["symbol"].astype(str) == symbol].reset_index(drop=True)
        window = _build_actual_window(
            symbol_raw=symbol_raw,
            symbol=symbol,
            origin_date=origin_date,
            history_length=args.history_length,
            horizon=args.horizon,
        )
        endpoint_paths = _endpoint_prediction_paths(
            prediction_frame=prediction_frame,
            symbol=symbol,
            origin_date=origin_date,
            current_close=window.current_close,
            future_dates=window.future["date"].tolist(),
        )
        kronos_path = _kronos_prediction_path(
            predictor=kronos_predictor,
            symbol=symbol,
            symbol_raw=symbol_raw,
            origin_date=origin_date,
            context_length=int(kronos_settings["context_length"]),
            horizon=args.horizon,
            sample_count=int(kronos_settings["sample_count"]),
            temperature=float(kronos_settings["temperature"]),
            top_p=float(kronos_settings["top_p"]),
        )
        all_paths = endpoint_paths + [kronos_path]
        figure_path = _plot_forecast_window(
            window=window,
            prediction_paths=all_paths,
            output_path=figures_dir / symbol.lower() / f"{symbol.lower()}_forecast_window_{origin_date:%Y%m%d}.png",
        )
        figure_rows.append({"symbol": symbol, "figure": figure_path.as_posix()})
        for path in all_paths:
            forecast_rows.extend(_path_records(symbol=symbol, origin_date=origin_date, path=path, window=window))
            metric_rows.append(_path_metrics(symbol=symbol, origin_date=origin_date, path=path, window=window))

    forecasts = pd.DataFrame(forecast_rows)
    metrics = pd.DataFrame(metric_rows)
    figures = pd.DataFrame(figure_rows)
    forecasts_path = reports_dir / "forecast_window_paths.csv"
    metrics_path = reports_dir / "forecast_window_metrics.csv"
    figures_path = reports_dir / "forecast_window_figures.csv"
    forecasts.to_csv(forecasts_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    figures.to_csv(figures_path, index=False)
    report_path = _write_report(
        symbols=args.symbols,
        origin_date=origin_date,
        history_length=args.history_length,
        horizon=args.horizon,
        kronos_sample_count=args.sample_count,
        seed=args.seed,
        metrics=metrics,
        figures=figures,
        forecasts_path=forecasts_path,
        metrics_path=metrics_path,
        figures_path=figures_path,
        report_path=reports_dir / "forecast_window_summary.md",
    )
    logger.info("Forecast-window visualization complete: report=%s", report_path)


class ActualWindow:
    def __init__(self, *, symbol: str, origin_date: pd.Timestamp, history: pd.DataFrame, future: pd.DataFrame):
        self.symbol = symbol
        self.origin_date = origin_date
        self.history = history
        self.future = future
        self.current_close = float(history["close"].iloc[-1])


def _clean_output_dir(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if cwd not in resolved.parents:
        raise ValueError(f"Refusing to clean path outside workspace: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_origin_date(
    *,
    prediction_frame: pd.DataFrame,
    raw_panel: pd.DataFrame,
    symbols: list[str],
    requested_origin: str,
    horizon: int,
) -> pd.Timestamp:
    if requested_origin != "latest":
        return pd.Timestamp(requested_origin)

    common_dates: set[pd.Timestamp] | None = None
    for symbol in symbols:
        model_dates: set[pd.Timestamp] | None = None
        for _, group in prediction_frame.loc[prediction_frame["symbol"].astype(str) == symbol].groupby("model_family"):
            dates = set(pd.to_datetime(group["date"]))
            model_dates = dates if model_dates is None else model_dates & dates
        symbol_raw = raw_panel.loc[raw_panel["symbol"].astype(str) == symbol].reset_index(drop=True)
        valid_dates = set()
        for idx, date in enumerate(pd.to_datetime(symbol_raw["date"])):
            if idx + horizon < len(symbol_raw):
                valid_dates.add(date)
        model_dates = (model_dates or set()) & valid_dates
        common_dates = model_dates if common_dates is None else common_dates & model_dates

    if not common_dates:
        raise ValueError("Could not find a common origin date with predictions and future actual data.")
    return max(common_dates)


def _build_kronos_predictor(args: argparse.Namespace):
    config = read_yaml(args.kronos_config)
    context_length = int(config.get("context_length", 64))
    temperature = args.temperature if args.temperature is not None else float(config.get("temperature", 1.0))
    top_p = args.top_p if args.top_p is not None else float(config.get("top_p", 0.9))
    sample_count = int(args.sample_count if args.sample_count is not None else config.get("sample_count", 1))
    model_name = str(config.get("pretrained_model", "NeoQuasar/Kronos-small"))
    tokenizer_name = str(config.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base"))

    Kronos, KronosTokenizer, KronosPredictor = _load_kronos_classes(args.kronos_repo)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=context_length)
    return predictor, {
        "context_length": context_length,
        "temperature": temperature,
        "top_p": top_p,
        "sample_count": sample_count,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
    }


def _build_actual_window(
    *,
    symbol_raw: pd.DataFrame,
    symbol: str,
    origin_date: pd.Timestamp,
    history_length: int,
    horizon: int,
) -> ActualWindow:
    matches = symbol_raw.index[pd.to_datetime(symbol_raw["date"]) == origin_date].tolist()
    if not matches:
        raise ValueError(f"Origin date {origin_date.date()} not found for {symbol}.")
    origin_idx = matches[0]
    if origin_idx + horizon >= len(symbol_raw):
        raise ValueError(f"Not enough future rows after {origin_date.date()} for {symbol}.")
    history = symbol_raw.iloc[max(0, origin_idx - history_length + 1) : origin_idx + 1].copy()
    future = symbol_raw.iloc[origin_idx + 1 : origin_idx + horizon + 1].copy()
    return ActualWindow(symbol=symbol, origin_date=origin_date, history=history, future=future)


def _endpoint_prediction_paths(
    *,
    prediction_frame: pd.DataFrame,
    symbol: str,
    origin_date: pd.Timestamp,
    current_close: float,
    future_dates: list[pd.Timestamp],
) -> list[dict[str, Any]]:
    paths = []
    for model in ENDPOINT_MODELS:
        row = prediction_frame.loc[
            (prediction_frame["model_family"].astype(str) == model)
            & (prediction_frame["symbol"].astype(str) == symbol)
            & (pd.to_datetime(prediction_frame["date"]) == origin_date)
        ]
        if row.empty:
            raise ValueError(f"Missing {model} prediction for {symbol} at {origin_date.date()}.")
        endpoint_close = current_close * (1.0 + float(row["y_pred"].iloc[0]))
        closes = [
            current_close + (endpoint_close - current_close) * (step / len(future_dates))
            for step in range(1, len(future_dates) + 1)
        ]
        paths.append(
            {
                "model_family": model,
                "path_type": "endpoint_interpolation",
                "dates": future_dates,
                "pred_close": closes,
                "pred_endpoint_close": endpoint_close,
                "y_pred": float(row["y_pred"].iloc[0]),
            }
        )
    return paths


def _kronos_prediction_path(
    *,
    predictor: Any,
    symbol: str,
    symbol_raw: pd.DataFrame,
    origin_date: pd.Timestamp,
    context_length: int,
    horizon: int,
    sample_count: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    matches = symbol_raw.index[pd.to_datetime(symbol_raw["date"]) == origin_date].tolist()
    if not matches:
        raise ValueError(f"Origin date {origin_date.date()} not found for {symbol}.")
    origin_idx = matches[0]
    if origin_idx < context_length - 1:
        raise ValueError(f"Not enough Kronos context for {symbol} at {origin_date.date()}.")
    if origin_idx + horizon >= len(symbol_raw):
        raise ValueError(f"Not enough future timestamps for Kronos on {symbol}.")

    context = symbol_raw.iloc[origin_idx - context_length + 1 : origin_idx + 1].copy()
    future = symbol_raw.iloc[origin_idx + 1 : origin_idx + horizon + 1].copy()
    pred_df = predictor.predict(
        df=context[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
        x_timestamp=context["date"].reset_index(drop=True),
        y_timestamp=future["date"].reset_index(drop=True),
        pred_len=horizon,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    return {
        "model_family": "kronos",
        "path_type": "native_5_step",
        "dates": pd.to_datetime(list(pred_df.index)),
        "pred_close": [float(value) for value in pred_df["close"].tolist()],
        "pred_endpoint_close": float(pred_df["close"].iloc[-1]),
        "y_pred": float(pred_df["close"].iloc[-1]) / float(context["close"].iloc[-1]) - 1.0,
    }


def _plot_forecast_window(
    *,
    window: ActualWindow,
    prediction_paths: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual = pd.concat([window.history[["date", "close"]], window.future[["date", "close"]]], ignore_index=True)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(actual["date"], actual["close"], color="#1d4ed8", linewidth=1.8, label="Ground Truth")
    ax.axvline(window.origin_date, color="#111827", linestyle=":", linewidth=1.0, alpha=0.75, label="Forecast origin")

    for path in prediction_paths:
        model = str(path["model_family"])
        dates = [window.origin_date] + list(pd.to_datetime(path["dates"]))
        closes = [window.current_close] + [float(value) for value in path["pred_close"]]
        linestyle = "-" if model == "kronos" else "--"
        marker = "D" if model == "kronos" else "o"
        label = model
        if path["path_type"] == "endpoint_interpolation":
            label = f"{model} endpoint interp"
        ax.plot(
            dates,
            closes,
            color=MODEL_COLORS.get(model),
            linestyle=linestyle,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            label=label,
        )

    ax.set_title(f"{window.symbol}: 5-session forecast window from {window.origin_date:%Y-%m-%d}")
    ax.set_ylabel("Close price")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.35)
    ax.legend(title="", loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _path_records(
    *,
    symbol: str,
    origin_date: pd.Timestamp,
    path: dict[str, Any],
    window: ActualWindow,
) -> list[dict[str, Any]]:
    actual_by_date = {
        pd.Timestamp(row["date"]): float(row["close"])
        for _, row in window.future.iterrows()
    }
    rows = []
    for idx, (date, pred_close) in enumerate(zip(pd.to_datetime(path["dates"]), path["pred_close"]), start=1):
        actual_close = actual_by_date[pd.Timestamp(date)]
        rows.append(
            {
                "symbol": symbol,
                "origin_date": origin_date,
                "model_family": path["model_family"],
                "path_type": path["path_type"],
                "step": idx,
                "forecast_date": date,
                "actual_close": actual_close,
                "pred_close": float(pred_close),
                "close_error": float(pred_close) - actual_close,
            }
        )
    return rows


def _path_metrics(
    *,
    symbol: str,
    origin_date: pd.Timestamp,
    path: dict[str, Any],
    window: ActualWindow,
) -> dict[str, Any]:
    actual = window.future["close"].astype(float).to_list()
    pred = [float(value) for value in path["pred_close"]]
    errors = [pred_value - actual_value for pred_value, actual_value in zip(pred, actual)]
    endpoint_error = pred[-1] - actual[-1]
    actual_ret_5d = actual[-1] / window.current_close - 1.0
    pred_ret_5d = pred[-1] / window.current_close - 1.0
    return {
        "symbol": symbol,
        "origin_date": origin_date,
        "model_family": path["model_family"],
        "path_type": path["path_type"],
        "path_mae": sum(abs(error) for error in errors) / len(errors),
        "endpoint_abs_error": abs(endpoint_error),
        "actual_ret_5d": actual_ret_5d,
        "pred_ret_5d": pred_ret_5d,
        "direction_correct": (actual_ret_5d >= 0) == (pred_ret_5d >= 0),
    }


def _write_report(
    *,
    symbols: list[str],
    origin_date: pd.Timestamp,
    history_length: int,
    horizon: int,
    kronos_sample_count: int,
    seed: int,
    metrics: pd.DataFrame,
    figures: pd.DataFrame,
    forecasts_path: Path,
    metrics_path: Path,
    figures_path: Path,
    report_path: Path,
) -> Path:
    lines = [
        "# Forecast Window Summary",
        "",
        f"- Symbols: `{', '.join(symbols)}`",
        f"- Origin date: `{origin_date:%Y-%m-%d}`",
        f"- History length: `{history_length}` sessions",
        f"- Horizon: `{horizon}` sessions",
        f"- Kronos sample_count: `{kronos_sample_count}`",
        f"- Seed: `{seed}`",
        f"- Forecast path CSV: `{forecasts_path.as_posix()}`",
        f"- Metrics CSV: `{metrics_path.as_posix()}`",
        f"- Figure paths CSV: `{figures_path.as_posix()}`",
        "",
        "## Figures",
        "",
        _markdown_table(figures),
        "",
        "## Metrics",
        "",
        _markdown_table(metrics),
        "",
        "## Interpretation Notes",
        "",
        "`kronos` is a native 5-step price path from the foundation model. "
        "`xlstm_ts` and `itransformer` are trained as `target_ret_5d` endpoint models, "
        "so their 5-step lines are linear interpolation from the origin close to the predicted `t+5` close. "
        "Use their endpoint error and 5-day direction, not the interpolated path shape, as the real evaluation.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = [columns, ["---"] * len(columns)]
    for _, record in frame.iterrows():
        row = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                row.append("nan" if math.isnan(value) else f"{value:.6f}")
            else:
                row.append(str(value))
        rows.append(row)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    main()
