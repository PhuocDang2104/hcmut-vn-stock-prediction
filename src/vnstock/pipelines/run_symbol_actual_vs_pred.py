from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vnstock.evaluation.compare import load_all_predictions
from vnstock.utils.io import ensure_dir
from vnstock.utils.logging import get_logger


DEFAULT_MODELS = ["xlstm_ts", "itransformer", "kronos"]
DEFAULT_SYMBOLS = ["VIC", "FPT", "VHM"]
SYMBOL_ALIASES = {"VIN": "VIC"}
MODEL_COLORS = {
    "xlstm_ts": "#2563eb",
    "itransformer": "#f97316",
    "kronos": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot symbol-level actual vs predicted outputs.")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--feature-panel", default="data/processed/shared/feature_panel.parquet")
    parser.add_argument("--figures-dir", default="outputs/figures/model_compare/symbol_level")
    parser.add_argument("--reports-dir", default="outputs/reports/model_compare/symbol_level")
    parser.add_argument("--split", default="test")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_symbol_actual_vs_pred")
    figures_dir = Path(args.figures_dir)
    reports_dir = Path(args.reports_dir)
    if args.clean:
        _clean_output_dir(figures_dir)
        _clean_output_dir(reports_dir)
    figures_dir = ensure_dir(figures_dir)
    reports_dir = ensure_dir(reports_dir)

    predictions = load_all_predictions(args.predictions_dir)
    if predictions.empty:
        raise ValueError(f"No predictions found in {args.predictions_dir}.")
    predictions = predictions.loc[predictions["model_family"].isin(args.models)].copy()
    if args.split:
        predictions = predictions.loc[predictions["split"].astype(str) == args.split].copy()
    if predictions.empty:
        raise ValueError("No predictions found for the requested models/split.")

    symbols, symbol_notes = _resolve_symbols(predictions, args.symbols)
    panel = pd.read_parquet(args.feature_panel, columns=["symbol", "date", "close"])
    frame = predictions.loc[predictions["symbol"].astype(str).isin(symbols)].copy()
    frame = frame.merge(panel, on=["symbol", "date"], how="left")
    frame["actual_close_t_plus_5"] = frame["close"] * (1.0 + frame["y_true"])
    frame["pred_close_t_plus_5"] = frame["close"] * (1.0 + frame["y_pred"])
    frame = frame.sort_values(["symbol", "date", "model_family"])

    rows = []
    figure_rows = []
    for symbol in symbols:
        subset = frame.loc[frame["symbol"].astype(str) == symbol].copy()
        if subset.empty:
            continue
        symbol_dir = ensure_dir(figures_dir / symbol.lower())
        return_path = _plot_symbol(
            subset,
            symbol=symbol,
            actual_col="y_true",
            pred_col="y_pred",
            ylabel="5-day return",
            title=f"{symbol}: actual vs predicted 5-day return",
            output_path=symbol_dir / f"{symbol.lower()}_actual_vs_pred_return.png",
        )
        close_path = _plot_symbol(
            subset,
            symbol=symbol,
            actual_col="actual_close_t_plus_5",
            pred_col="pred_close_t_plus_5",
            ylabel="close at t+5",
            title=f"{symbol}: actual vs predicted close at t+5",
            output_path=symbol_dir / f"{symbol.lower()}_actual_vs_pred_close_tplus5.png",
        )
        figure_rows.extend(
            [
                {"symbol": symbol, "chart": "return", "path": return_path.as_posix()},
                {"symbol": symbol, "chart": "close_tplus5", "path": close_path.as_posix()},
            ]
        )
        rows.extend(_symbol_metrics(subset, symbol=symbol))

    metrics = pd.DataFrame(rows)
    figures = pd.DataFrame(figure_rows)
    metrics_path = reports_dir / "symbol_metrics.csv"
    figures_path = reports_dir / "figure_paths.csv"
    metrics.to_csv(metrics_path, index=False)
    figures.to_csv(figures_path, index=False)
    report_path = _write_report(
        symbols=symbols,
        symbol_notes=symbol_notes,
        metrics=metrics,
        figures=figures,
        report_path=reports_dir / "symbol_actual_vs_pred_summary.md",
        metrics_path=metrics_path,
        figures_path=figures_path,
    )
    logger.info("Symbol actual-vs-predicted complete: report=%s", report_path)


def _clean_output_dir(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if cwd not in resolved.parents:
        raise ValueError(f"Refusing to clean path outside workspace: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _resolve_symbols(predictions: pd.DataFrame, requested_symbols: list[str]) -> tuple[list[str], list[str]]:
    available = sorted(predictions["symbol"].dropna().astype(str).unique())
    lookup = {symbol.upper(): symbol for symbol in available}
    resolved = []
    notes = []
    for requested in requested_symbols:
        key = requested.upper()
        symbol = lookup.get(key)
        if symbol is None and key in SYMBOL_ALIASES:
            alias = SYMBOL_ALIASES[key]
            symbol = lookup.get(alias.upper())
            if symbol:
                notes.append(f"`{requested}` not found; resolved to `{symbol}`.")
        if symbol is None:
            raise ValueError(f"Symbol `{requested}` not found in predictions.")
        if symbol not in resolved:
            resolved.append(symbol)
    return resolved, notes


def _plot_symbol(
    subset: pd.DataFrame,
    *,
    symbol: str,
    actual_col: str,
    pred_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    actual = subset[["date", actual_col]].drop_duplicates("date").sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(actual["date"], actual[actual_col], color="#111827", linewidth=1.8, label="actual")

    for model, group in subset.groupby("model_family", sort=False):
        group = group.sort_values("date")
        color = MODEL_COLORS.get(str(model), None)
        label = f"{model} pred"
        if len(group) <= 2:
            ax.scatter(
                group["date"],
                group[pred_col],
                color=color,
                s=110,
                marker="D",
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
                label=f"{label} ({len(group)} point)",
            )
            for _, record in group.iterrows():
                ax.annotate(
                    str(model),
                    (record["date"], record[pred_col]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
        else:
            ax.plot(group["date"], group[pred_col], color=color, linewidth=1.35, alpha=0.9, label=label)

    if "return" in ylabel:
        ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("prediction date t")
    ax.legend(title="", loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _symbol_metrics(subset: pd.DataFrame, *, symbol: str) -> list[dict[str, object]]:
    rows = []
    for model, group in subset.groupby("model_family"):
        error = group["y_pred"] - group["y_true"]
        price_error = group["pred_close_t_plus_5"] - group["actual_close_t_plus_5"]
        rank_ic = group["y_pred"].corr(group["y_true"], method="spearman") if len(group) > 2 else math.nan
        rows.append(
            {
                "symbol": symbol,
                "model_family": model,
                "rows": len(group),
                "date_min": group["date"].min(),
                "date_max": group["date"].max(),
                "return_mae": error.abs().mean(),
                "return_rmse": math.sqrt((error**2).mean()),
                "close_mae": price_error.abs().mean(),
                "directional_accuracy": ((group["y_pred"] >= 0) == (group["y_true"] >= 0)).mean(),
                "rank_ic": rank_ic,
            }
        )
    return rows


def _write_report(
    *,
    symbols: list[str],
    symbol_notes: list[str],
    metrics: pd.DataFrame,
    figures: pd.DataFrame,
    report_path: Path,
    metrics_path: Path,
    figures_path: Path,
) -> Path:
    lines = [
        "# Symbol Actual vs Predicted Summary",
        "",
        f"- Symbols: `{', '.join(symbols)}`",
        f"- Metrics CSV: `{metrics_path.as_posix()}`",
        f"- Figure paths CSV: `{figures_path.as_posix()}`",
    ]
    for note in symbol_notes:
        lines.append(f"- Symbol note: {note}")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            _markdown_table(metrics),
            "",
            "## Figures",
            "",
            _markdown_table(figures),
            "",
            "## Metric Guidance",
            "",
            "Use `rank_ic`, `top-k realized return`, and `long-short spread` for stock selection. "
            "Use `directional_accuracy` only as a secondary sanity check, because near-zero returns "
            "and one-sided market regimes can inflate or depress accuracy.",
            "",
            "Kronos currently appears as a single diamond marker per symbol because the present "
            "zero-shot output was generated with `latest_only = true`, so it has one test date per symbol.",
        ]
    )
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
