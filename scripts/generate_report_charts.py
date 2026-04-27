from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vnstock.pipelines.run_investment_backtest import _prepare_frame  # noqa: E402
from vnstock.utils.io import ensure_dir, load_table  # noqa: E402


FIGURE_DIR = ROOT / "outputs" / "figures" / "report_charts"
REPORT_DIR = ROOT / "outputs" / "reports" / "report_charts"
FEATURE_PANEL = ROOT / "data" / "processed" / "shared" / "feature_panel.parquet"
MODEL_METRICS = ROOT / "outputs" / "reports" / "final_top5_model_suite" / "top5_model_suite_metrics.csv"
PREDICTIONS = ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"
EQUITY_CURVE = ROOT / "outputs" / "reports" / "investment_application_10m" / "fixed_5day_rebalance_equity.csv"
SCORE_BUCKETS = ROOT / "outputs" / "reports" / "investment_application_10m" / "score_bucket_returns.csv"


COLORS = {
    "hybrid": "#0B1F3A",
    "accent": "#E76F51",
    "blue": "#2A6FBB",
    "green": "#2A9D8F",
    "gold": "#E9C46A",
    "gray": "#6C757D",
    "light": "#EEF2F5",
}


def main() -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(REPORT_DIR)

    feature_panel = load_table(FEATURE_PANEL)
    metrics = pd.read_csv(MODEL_METRICS)
    pred = _prepare_frame(load_table(PREDICTIONS), split="test", score_column="y_pred")
    daily = compute_daily_metrics(pred)
    equity = pd.read_csv(EQUITY_CURVE, parse_dates=["date"])
    buckets = pd.read_csv(SCORE_BUCKETS)

    generated = [
        plot_data_split_timeline(feature_panel),
        plot_model_comparison(metrics),
        plot_top5_vs_longshort(metrics),
        plot_daily_ic(daily),
        plot_score_buckets(buckets),
        plot_equity_curve(equity),
        plot_monthly_stability(daily),
        plot_drawdown_curve(equity),
        plot_score_distribution(pred),
    ]
    write_index(generated)
    print(FIGURE_DIR)
    print(REPORT_DIR / "report_chart_index.md")


def compute_daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date"):
        clean = group.dropna(subset=["y_true", "y_pred"]).copy()
        if len(clean) < 2:
            continue
        ic = _safe_corr(clean["y_true"].to_numpy(float), clean["y_pred"].to_numpy(float))
        rankic = _safe_corr(
            clean["y_true"].rank(method="average").to_numpy(float),
            clean["y_pred"].rank(method="average").to_numpy(float),
        )
        ranked = clean.sort_values("y_pred", ascending=False)
        top = ranked.head(min(5, len(ranked)))
        bottom = ranked.tail(min(5, len(ranked)))
        rows.append(
            {
                "date": pd.Timestamp(date),
                "IC": ic,
                "RankIC": rankic,
                "Top5_Return": float(top["y_true"].mean()),
                "Bottom5_Return": float(bottom["y_true"].mean()),
                "LongShort5": float(top["y_true"].mean() - bottom["y_true"].mean()),
                "Top5_Direction_Acc": float((top["y_true"] > 0).mean()),
            }
        )
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["IC_20d"] = daily["IC"].rolling(20, min_periods=5).mean()
    daily["RankIC_20d"] = daily["RankIC"].rolling(20, min_periods=5).mean()
    daily["Top5_Return_20d"] = daily["Top5_Return"].rolling(20, min_periods=5).mean()
    daily["LongShort5_20d"] = daily["LongShort5"].rolling(20, min_periods=5).mean()
    daily.to_csv(REPORT_DIR / "daily_hybrid_metrics.csv", index=False)
    return daily


def plot_data_split_timeline(feature_panel: pd.DataFrame) -> Path:
    frame = feature_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    order = ["train", "valid", "test"]
    summary = (
        frame.groupby("split")
        .agg(start=("date", "min"), end=("date", "max"), rows=("date", "size"), symbols=("symbol", "nunique"))
        .reindex(order)
        .dropna()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 3.8))
    colors = {"train": COLORS["blue"], "valid": COLORS["gold"], "test": COLORS["accent"]}
    for idx, row in summary.iterrows():
        start = mdates.date2num(row["start"])
        end = mdates.date2num(row["end"])
        ax.barh(idx, end - start, left=start, height=0.52, color=colors[row["split"]], edgecolor="white")
        label = f"{row['split']} | {row['start'].date()} to {row['end'].date()} | rows={int(row['rows']):,}"
        ax.text(start + (end - start) / 2, idx, label, ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary["split"].str.upper())
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Chronological Data Split Timeline", fontsize=14, weight="bold")
    ax.set_xlabel("Date")
    ax.grid(axis="x", alpha=0.25)
    return save_fig(fig, "01_data_split_timeline.png")


def plot_model_comparison(metrics: pd.DataFrame) -> Path:
    frame = metrics.copy()
    frame["model_short"] = frame["model"].map(short_model_name)
    columns = ["IC", "RankIC", "Top5_Return", "LongShort5"]
    values = frame[columns].to_numpy(float) * 100.0
    x = np.arange(len(frame))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 5.5))
    palette = [COLORS["hybrid"], COLORS["blue"], COLORS["green"], COLORS["accent"]]
    for idx, column in enumerate(columns):
        ax.bar(x + (idx - 1.5) * width, values[:, idx], width, label=column, color=palette[idx], alpha=0.92)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["model_short"], rotation=18, ha="right")
    ax.set_ylabel("Metric value (%)")
    ax.set_title("Model Comparison: Ranking And Portfolio Metrics", fontsize=14, weight="bold")
    ax.legend(ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    annotate_best(ax, x, values[:, 0])
    return save_fig(fig, "02_model_comparison_metrics.png")


def plot_top5_vs_longshort(metrics: pd.DataFrame) -> Path:
    frame = metrics.copy()
    frame["model_short"] = frame["model"].map(short_model_name)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    x = frame["Top5_Return"] * 100.0
    y = frame["LongShort5"] * 100.0
    size = 250 + frame["RankIC"].clip(lower=0) * 5000
    colors = [COLORS["hybrid"] if "Hybrid" in name else COLORS["blue"] for name in frame["model"]]
    ax.scatter(x, y, s=size, c=colors, alpha=0.82, edgecolor="white", linewidth=1.2)
    for _, row in frame.iterrows():
        ax.annotate(short_model_name(row["model"]), (row["Top5_Return"] * 100, row["LongShort5"] * 100), xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xlabel("Top5 Return per 5-session window (%)")
    ax.set_ylabel("LongShort5 spread (%)")
    ax.set_title("Top5 Return vs LongShort5", fontsize=14, weight="bold")
    ax.grid(alpha=0.25)
    return save_fig(fig, "03_top5_return_vs_longshort.png")


def plot_daily_ic(daily: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(daily["date"], daily["IC"], color=COLORS["light"], edgecolor=COLORS["gray"], linewidth=0.2, label="Daily IC")
    ax.plot(daily["date"], daily["IC_20d"], color=COLORS["accent"], linewidth=2.2, label="20-day rolling IC")
    ax.plot(daily["date"], daily["RankIC_20d"], color=COLORS["hybrid"], linewidth=2.0, label="20-day rolling RankIC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily IC And Rolling Ranking Stability", fontsize=14, weight="bold")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    return save_fig(fig, "04_daily_ic_rolling_ic.png")


def plot_score_buckets(buckets: pd.DataFrame) -> Path:
    frame = buckets.copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = frame["score_bucket"].astype(int)
    ax.bar(x, frame["avg_realized_return"] * 100.0, color=COLORS["blue"], alpha=0.9, label="Avg realized return")
    ax.set_xlabel("Score bucket (1 = lowest, 5 = highest)")
    ax.set_ylabel("Avg realized return (%)", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax2 = ax.twinx()
    ax2.plot(x, frame["direction_acc"] * 100.0, marker="o", color=COLORS["accent"], linewidth=2.4, label="Direction accuracy")
    ax2.set_ylabel("Direction accuracy (%)", color=COLORS["accent"])
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax.set_title("Score Buckets: Higher Score, Higher Realized Return", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    return save_fig(fig, "05_score_bucket_returns.png")


def plot_equity_curve(equity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(equity["date"], equity["equity"] / 1_000_000, color=COLORS["hybrid"], linewidth=2.5, label="Hybrid top-5 portfolio")
    ax.plot(equity["date"], equity["benchmark_equity"] / 1_000_000, color=COLORS["gray"], linewidth=2.0, linestyle="--", label="Equal-weight universe benchmark")
    ax.set_title("10M VND Backtest Equity Curve", fontsize=14, weight="bold")
    ax.set_ylabel("Capital (million VND)")
    ax.set_xlabel("Rebalance date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return save_fig(fig, "06_equity_curve_10m.png")


def plot_monthly_stability(daily: pd.DataFrame) -> Path:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        frame.groupby("month", as_index=False)
        .agg(
            Top5_Return=("Top5_Return", "mean"),
            LongShort5=("LongShort5", "mean"),
            Top5_Direction_Acc=("Top5_Direction_Acc", "mean"),
            IC=("IC", "mean"),
        )
        .sort_values("month")
    )
    monthly.to_csv(REPORT_DIR / "monthly_hybrid_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    colors = np.where(monthly["Top5_Return"] >= 0, COLORS["green"], COLORS["accent"])
    ax.bar(monthly["month"], monthly["Top5_Return"] * 100.0, width=22, color=colors, alpha=0.9, label="Monthly avg Top5 Return")
    ax.plot(monthly["month"], monthly["IC"], color=COLORS["hybrid"], linewidth=2.1, marker="o", label="Monthly avg IC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Stability: Top5 Return And IC", fontsize=14, weight="bold")
    ax.set_ylabel("Top5 return (%) / IC")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    return save_fig(fig, "07_monthly_top5_stability.png")


def plot_drawdown_curve(equity: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.fill_between(equity["date"], equity["drawdown"] * 100.0, 0, color=COLORS["accent"], alpha=0.35)
    ax.plot(equity["date"], equity["drawdown"] * 100.0, color=COLORS["accent"], linewidth=1.8)
    ax.set_title("10M VND Backtest Drawdown", fontsize=14, weight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Rebalance date")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return save_fig(fig, "08_drawdown_curve_10m.png")


def plot_score_distribution(frame: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.hist(frame["y_pred"], bins=70, color=COLORS["hybrid"], alpha=0.82)
    ax.axvline(frame["y_pred"].median(), color=COLORS["gold"], linewidth=2.4, label="Median score")
    ax.axvline(frame["y_pred"].quantile(0.95), color=COLORS["accent"], linewidth=2.4, linestyle="--", label="95th percentile")
    ax.set_title("Prediction Score Distribution On Test Set", fontsize=14, weight="bold")
    ax.set_xlabel("Hybrid score")
    ax.set_ylabel("Rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return save_fig(fig, "09_prediction_score_distribution.png")


def write_index(paths: list[Path]) -> None:
    descriptions = {
        "01_data_split_timeline.png": "Chronological train, validation, and test split.",
        "02_model_comparison_metrics.png": "IC, RankIC, Top5 Return, and LongShort5 across models.",
        "03_top5_return_vs_longshort.png": "Portfolio-level model trade-off between top5 return and long-short spread.",
        "04_daily_ic_rolling_ic.png": "Daily IC plus rolling IC/RankIC stability over time.",
        "05_score_bucket_returns.png": "Monotonic relationship between score bucket and realized return.",
        "06_equity_curve_10m.png": "10M VND long-only top-5 equity curve vs benchmark.",
        "07_monthly_top5_stability.png": "Monthly top5 return and IC stability for paper/report discussion.",
        "08_drawdown_curve_10m.png": "Portfolio drawdown path for risk discussion.",
        "09_prediction_score_distribution.png": "Distribution of model scores on the test set.",
    }
    rows = [
        "# Report Chart Pack",
        "",
        f"Figure directory: `{FIGURE_DIR.as_posix()}`",
        "",
        "| Figure | Purpose |",
        "| --- | --- |",
    ]
    for path in paths:
        rows.append(f"| `{path.name}` | {descriptions.get(path.name, '')} |")
    rows.extend(
        [
            "",
            "Generated data tables:",
            "",
            f"- `{(REPORT_DIR / 'daily_hybrid_metrics.csv').as_posix()}`",
            f"- `{(REPORT_DIR / 'monthly_hybrid_metrics.csv').as_posix()}`",
            "",
            "Regenerate:",
            "",
            "```powershell",
            "python scripts\\generate_report_charts.py",
            "```",
        ]
    )
    (REPORT_DIR / "report_chart_index.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def short_model_name(name: str) -> str:
    mapping = {
        "Hybrid xLSTM Direction-Excess Blend": "Hybrid xLSTM",
        "LightGBM-style HGBR": "LightGBM/HGBR",
        "CNN-LSTM": "CNN-LSTM",
        "TCN": "TCN",
        "PatchTST": "PatchTST",
        "Kronos zero-shot": "Kronos",
    }
    return mapping.get(str(name), str(name))


def annotate_best(ax: plt.Axes, x: np.ndarray, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    idx = int(np.nanargmax(values))
    ax.text(x[idx] - 0.27, values[idx] + 0.18, "best IC", fontsize=8, color=COLORS["hybrid"], weight="bold")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    main()
