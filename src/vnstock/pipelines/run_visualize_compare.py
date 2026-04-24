from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vnstock.evaluation.compare import load_all_predictions
from vnstock.evaluation.leaderboard import build_leaderboard
from vnstock.utils.io import ensure_dir, save_table
from vnstock.utils.logging import get_logger


DEFAULT_MODELS = ["xlstm_ts", "itransformer", "kronos"]
DEFAULT_SYMBOL_ALIASES = {
    "VIN": "VIC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize model prediction comparison outputs.")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--figures-dir", default="outputs/figures/model_compare")
    parser.add_argument("--reports-dir", default="outputs/reports/model_compare")
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--symbol", default="FPT")
    parser.add_argument("--align-intersection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_visualize_compare")
    figures_dir = ensure_dir(args.figures_dir)
    reports_dir = ensure_dir(args.reports_dir)

    predictions = load_all_predictions(args.predictions_dir)
    predictions = predictions.loc[predictions["model_family"].isin(args.models)].copy()
    if args.split:
        predictions = predictions.loc[predictions["split"].astype(str) == args.split].copy()
    if args.align_intersection:
        predictions = _filter_model_intersection(predictions)
    if predictions.empty:
        raise ValueError("No predictions found for the requested models/split.")

    leaderboard = build_leaderboard(
        args.predictions_dir,
        split=args.split,
        align_intersection=args.align_intersection,
    )
    leaderboard = leaderboard.loc[leaderboard["model_family"].isin(args.models)].copy()
    leaderboard_path = save_table(leaderboard, reports_dir / "leaderboard.csv")
    resolved_symbol, symbol_note = _resolve_symbol(
        predictions,
        requested_symbol=args.symbol,
        aliases=DEFAULT_SYMBOL_ALIASES,
    )

    sns.set_theme(style="whitegrid")
    figure_paths = {
        "leaderboard_huber": _plot_metric_bar(
            leaderboard,
            metric="huber",
            output_path=figures_dir / "leaderboard_huber.png",
        ),
        "leaderboard_ic": _plot_metric_bar(
            leaderboard,
            metric="information_coefficient",
            output_path=figures_dir / "leaderboard_ic.png",
            ascending=False,
        ),
        "prediction_scatter": _plot_prediction_scatter(
            predictions,
            output_path=figures_dir / "prediction_scatter.png",
        ),
        "residual_distribution": _plot_residual_distribution(
            predictions,
            output_path=figures_dir / "residual_distribution.png",
        ),
        "daily_ic": _plot_daily_ic(
            predictions,
            output_path=figures_dir / "daily_ic_rolling20.png",
        ),
        "symbol_trace": _plot_symbol_trace(
            predictions,
            symbol=resolved_symbol,
            output_path=figures_dir / f"{resolved_symbol.lower()}_actual_vs_pred.png",
        ),
    }

    report_path = _write_report(
        leaderboard=leaderboard,
        prediction_rows=len(predictions),
        split=args.split,
        models=sorted(predictions["model_family"].astype(str).unique()),
        requested_symbol=args.symbol,
        resolved_symbol=resolved_symbol,
        symbol_note=symbol_note,
        report_path=reports_dir / "benchmark_summary.md",
        figure_paths=figure_paths,
        leaderboard_path=leaderboard_path,
    )
    logger.info("Visualization complete: report=%s figures=%s", report_path, figure_paths)


def _resolve_symbol(
    predictions: pd.DataFrame,
    *,
    requested_symbol: str,
    aliases: dict[str, str],
) -> tuple[str, str | None]:
    symbols = sorted(predictions["symbol"].dropna().astype(str).unique())
    symbol_lookup = {symbol.upper(): symbol for symbol in symbols}
    requested_key = requested_symbol.upper()

    if requested_key in symbol_lookup:
        return symbol_lookup[requested_key], None

    alias = aliases.get(requested_key)
    if alias and alias.upper() in symbol_lookup:
        resolved = symbol_lookup[alias.upper()]
        return resolved, f"Requested symbol `{requested_symbol}` is not in predictions; resolved to `{resolved}`."

    close_matches = [
        symbol for symbol in symbols if requested_key in symbol.upper() or symbol.upper() in requested_key
    ][:10]
    hint = f" Close matches: {', '.join(close_matches)}." if close_matches else ""
    raise ValueError(f"Symbol `{requested_symbol}` not found in predictions.{hint}")


def _plot_metric_bar(
    leaderboard: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    ascending: bool = True,
) -> Path:
    ordered = leaderboard.sort_values(metric, ascending=ascending)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=ordered, x=metric, y="model_family", hue="model_family", legend=False, ax=ax)
    ax.set_title(f"Model comparison by {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _filter_model_intersection(predictions: pd.DataFrame) -> pd.DataFrame:
    common_keys = None
    for _, frame in predictions.groupby("model_family"):
        keys = set(zip(frame["symbol"].astype(str), pd.to_datetime(frame["date"])))
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = common_keys or set()
    key_index = list(zip(predictions["symbol"].astype(str), pd.to_datetime(predictions["date"])))
    mask = [key in common_keys for key in key_index]
    return predictions.loc[mask].copy()


def _plot_prediction_scatter(predictions: pd.DataFrame, *, output_path: Path) -> Path:
    sample = predictions.sample(n=min(8000, len(predictions)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=sample,
        x="y_true",
        y="y_pred",
        hue="model_family",
        alpha=0.35,
        s=14,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Predicted vs realized 5-day return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_residual_distribution(predictions: pd.DataFrame, *, output_path: Path) -> Path:
    frame = predictions.copy()
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(data=frame, x="residual", hue="model_family", common_norm=False, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_daily_ic(predictions: pd.DataFrame, *, output_path: Path) -> Path:
    records = []
    for (model, date), group in predictions.groupby(["model_family", "date"]):
        if len(group) < 2:
            continue
        records.append(
            {
                "model_family": model,
                "date": date,
                "rank_ic": group["y_true"].corr(group["y_pred"], method="spearman"),
            }
        )
    ic_frame = pd.DataFrame(records).sort_values(["model_family", "date"])
    ic_frame["rank_ic_rolling20"] = ic_frame.groupby("model_family")["rank_ic"].transform(
        lambda series: series.rolling(20, min_periods=5).mean()
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.lineplot(data=ic_frame, x="date", y="rank_ic_rolling20", hue="model_family", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("20-day rolling Rank IC on test split")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_symbol_trace(predictions: pd.DataFrame, *, symbol: str, output_path: Path) -> Path:
    subset = predictions.loc[predictions["symbol"].astype(str) == symbol].copy()
    if subset.empty:
        raise ValueError(f"Symbol `{symbol}` not found in predictions.")
    subset = subset.sort_values(["date", "model_family"])
    actual = (
        subset[["date", "y_true"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(actual["date"], actual["y_true"], color="black", linewidth=1.6, label="actual")
    sns.lineplot(data=subset, x="date", y="y_pred", hue="model_family", marker="o", markersize=3, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Actual vs predicted 5-day return for {subset['symbol'].iloc[0]}")
    ax.set_ylabel("5-day return")
    ax.set_xlabel("")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _write_report(
    *,
    leaderboard: pd.DataFrame,
    prediction_rows: int,
    split: str,
    models: list[str],
    requested_symbol: str,
    resolved_symbol: str,
    symbol_note: str | None,
    report_path: Path,
    figure_paths: dict[str, Path],
    leaderboard_path: Path,
) -> Path:
    metric_columns = [
        "model_family",
        "rows",
        "huber",
        "mae",
        "rmse",
        "information_coefficient",
        "directional_accuracy",
        "confident_directional_accuracy_top20",
        "large_move_directional_accuracy_2pct",
        "direction_threshold",
        "direction_positive_rate",
        "top_k_realized_return",
        "long_short_spread",
    ]
    metric_columns = [column for column in metric_columns if column in leaderboard.columns]
    metric_table = _markdown_table(leaderboard[metric_columns])
    lines = [
        "# Model Benchmark",
        "",
        f"- Evaluated split: `{split}`",
        f"- Models: `{', '.join(models)}`",
        f"- Prediction rows used for visualization: `{prediction_rows}`",
        f"- Leaderboard CSV: `{leaderboard_path.as_posix()}`",
        f"- Requested symbol: `{requested_symbol}`",
        f"- Visualized symbol: `{resolved_symbol}`",
        f"- Symbol note: {symbol_note or 'n/a'}",
        "",
        "## Result Summary",
        "",
        metric_table,
        "",
        "Lower `huber`, `mae`, and `rmse` are better. Higher `information_coefficient`, "
        "`top_k_realized_return`, and `long_short_spread` are better.",
        "",
        "## Figures",
        "",
        f"- Huber leaderboard: `{figure_paths['leaderboard_huber'].as_posix()}`",
        f"- Rank IC leaderboard: `{figure_paths['leaderboard_ic'].as_posix()}`",
        f"- Prediction scatter: `{figure_paths['prediction_scatter'].as_posix()}`",
        f"- Residual distribution: `{figure_paths['residual_distribution'].as_posix()}`",
        f"- Rolling Rank IC: `{figure_paths['daily_ic'].as_posix()}`",
        f"- `{resolved_symbol}` actual vs predicted: `{figure_paths['symbol_trace'].as_posix()}`",
        "",
        "## How To Reproduce",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "python -m vnstock.pipelines.run_build_shared_dataset --config configs/data/dataset_daily.yaml --use-interim",
        "python -m vnstock.pipelines.run_xlstm_ts --config configs/models/xlstm_ts.yaml",
        "python -m vnstock.pipelines.run_itransformer --config configs/models/itransformer.yaml",
        "python -m vnstock.pipelines.run_kronos_zero_shot --config configs/models/kronos.yaml --kronos-repo external/Kronos --split test --device cpu",
        "python -m vnstock.pipelines.run_compare --predictions-dir outputs/predictions --leaderboard-output outputs/metrics/leaderboard_xlstm_itransformer.csv --combined-output outputs/predictions/all_predictions_xlstm_itransformer.parquet --split test",
        "python -m vnstock.pipelines.run_compare --predictions-dir outputs/predictions --leaderboard-output outputs/metrics/leaderboard_3models_aligned.csv --combined-output outputs/predictions/all_predictions_3models.parquet --split test --align-intersection",
        "python -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT",
        "python -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT --align-intersection",
        "```",
        "",
        "## Kronos Note",
        "",
        "Kronos is run through `python -m vnstock.pipelines.run_kronos_zero_shot` when the official "
        "Kronos repo/module is available. Use `--align-intersection` in compare/visualize when Kronos "
        "is inferred on a smaller sampled subset.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [columns, ["---"] * len(columns)]
    for _, record in frame.iterrows():
        row = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                row.append(f"{value:.6f}")
            else:
                row.append(str(value))
        rows.append(row)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


if __name__ == "__main__":
    main()
