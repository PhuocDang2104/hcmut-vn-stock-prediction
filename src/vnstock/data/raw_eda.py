from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnstock.data.loaders import load_raw_panel, load_universe
from vnstock.utils.io import ensure_dir, save_table
from vnstock.utils.paths import path_for, resolve_path


sns.set_theme(style="whitegrid")


def _active_universe_scope(config: dict[str, Any]) -> str:
    return str(config.get("raw_download", {}).get("universe_scope", "broad"))


def load_active_raw_panel(config: dict[str, Any]) -> pd.DataFrame:
    universe_scope = _active_universe_scope(config)
    universe_path = resolve_path(config["universes"][universe_scope])
    universe = load_universe(universe_path)
    panel = load_raw_panel(path_for("raw_root"), universe=universe)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def _chart_title_prefix(panel: pd.DataFrame) -> str:
    return f"Training Universe ({panel['symbol'].nunique()} symbols)"


def _heatmap_annot(symbol_count: int) -> bool:
    return symbol_count <= 20


def _rebased_figure_size(symbol_count: int) -> tuple[float, float]:
    return (16, 8 if symbol_count <= 30 else 9)


def _boxplot_figure_size(symbol_count: int) -> tuple[float, float]:
    return (max(12, min(32, symbol_count * 0.35)), 6.5)


def _correlation_figure_size(symbol_count: int) -> tuple[float, float]:
    side = max(10, min(28, symbol_count * 0.28))
    return (side, side)


def build_symbol_summary(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.sort_values(["symbol", "date"]).copy()
    enriched["ret_1d"] = enriched.groupby("symbol")["close"].pct_change()
    summary = (
        enriched.groupby("symbol")
        .agg(
            rows=("date", "size"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            close_first=("close", "first"),
            close_last=("close", "last"),
            volume_median=("volume", "median"),
            ret_1d_volatility=("ret_1d", "std"),
        )
        .reset_index()
        .sort_values("symbol")
    )
    summary["close_change_pct"] = summary["close_last"] / summary["close_first"] - 1.0
    return summary


def plot_rebased_close(panel: pd.DataFrame, output_path: Path) -> Path:
    pivot = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    rebased = pivot.divide(pivot.ffill().bfill().iloc[0]).mul(100)
    symbol_count = panel["symbol"].nunique()

    fig, ax = plt.subplots(figsize=_rebased_figure_size(symbol_count))
    rebased.plot(ax=ax, linewidth=1.4)
    ax.set_title(f"{_chart_title_prefix(panel)} Rebased Close (Base = 100)")
    ax.set_ylabel("Indexed Close")
    ax.set_xlabel("")
    if symbol_count <= 20:
        ax.legend(loc="upper left", ncol=2, frameon=True)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_volume_boxplot(panel: pd.DataFrame, output_path: Path) -> Path:
    frame = panel.copy()
    frame["log_volume"] = np.log10(frame["volume"].clip(lower=1).astype(float))
    symbol_count = panel["symbol"].nunique()

    fig, ax = plt.subplots(figsize=_boxplot_figure_size(symbol_count))
    sns.boxplot(data=frame, x="symbol", y="log_volume", ax=ax)
    ax.set_title(f"{_chart_title_prefix(panel)} Volume Distribution (log10)")
    ax.set_xlabel("")
    ax.set_ylabel("log10(volume)")
    if symbol_count > 20:
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_return_correlation(panel: pd.DataFrame, output_path: Path) -> Path:
    frame = panel.sort_values(["symbol", "date"]).copy()
    frame["ret_1d"] = frame.groupby("symbol")["close"].pct_change()
    pivot = frame.pivot(index="date", columns="symbol", values="ret_1d")
    corr = pivot.corr()
    symbol_count = panel["symbol"].nunique()

    fig, ax = plt.subplots(figsize=_correlation_figure_size(symbol_count))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=_heatmap_annot(symbol_count),
        fmt=".2f",
        ax=ax,
    )
    ax.set_title(f"{_chart_title_prefix(panel)} Daily Return Correlation")
    if symbol_count > 20:
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_monthly_coverage(panel: pd.DataFrame, output_path: Path) -> Path:
    frame = panel.copy()
    frame["year_month"] = frame["date"].dt.to_period("M").astype(str)
    coverage = (
        frame.groupby(["symbol", "year_month"])["date"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    symbol_count = panel["symbol"].nunique()
    width = max(24, min(40, coverage.shape[1] * 0.25))
    height = max(5, min(24, symbol_count * 0.25))

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(coverage, cmap="Greens", ax=ax)
    ax.set_title(f"{_chart_title_prefix(panel)} Monthly Trading Session Coverage")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_markdown_summary(panel: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> Path:
    date_min = pd.to_datetime(panel["date"]).min().date().isoformat()
    date_max = pd.to_datetime(panel["date"]).max().date().isoformat()
    title = _chart_title_prefix(panel)
    source_mix = ", ".join(
        f"{source}={count}"
        for source, count in panel.groupby("source")["symbol"].nunique().sort_index().items()
    )
    lines = [
        "# Raw Data Summary",
        "",
        f"- universe: {title}",
        f"- symbols: {panel['symbol'].nunique()}",
        f"- rows: {len(panel)}",
        f"- date range: {date_min} to {date_max}",
        f"- symbols by source: {source_mix}",
        "",
        "## Symbol Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_raw_eda(config: dict[str, Any]) -> dict[str, Path]:
    figures_root = ensure_dir(path_for("outputs_root") / "figures" / "raw_data")
    reports_root = ensure_dir(path_for("outputs_root") / "reports")
    metrics_root = ensure_dir(path_for("outputs_root") / "metrics")

    panel = load_active_raw_panel(config)
    summary = build_symbol_summary(panel)

    outputs = {
        "summary_csv": save_table(summary, metrics_root / "raw_symbol_summary.csv"),
        "rebased_close": plot_rebased_close(panel, figures_root / "raw_universe_rebased_close.png"),
        "volume_boxplot": plot_volume_boxplot(panel, figures_root / "raw_universe_volume_boxplot.png"),
        "return_corr": plot_return_correlation(panel, figures_root / "raw_universe_return_correlation.png"),
        "coverage_heatmap": plot_monthly_coverage(panel, figures_root / "raw_universe_monthly_coverage.png"),
        "summary_md": write_markdown_summary(panel, summary, reports_root / "raw_data_summary.md"),
    }
    return outputs
