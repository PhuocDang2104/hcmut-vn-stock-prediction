from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vnstock.utils.io import ensure_dir, save_table
from vnstock.utils.paths import path_for


sns.set_theme(style="whitegrid")


def _raw_data_root() -> Path:
    return path_for("raw_root").parent


def _figures_root() -> Path:
    return ensure_dir(_raw_data_root() / "_viz")


def _list_data_files(folder: Path) -> list[Path]:
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix in {".csv", ".json"}])


def _load_csv_folder(folder: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _list_data_files(folder):
        if path.suffix != ".csv" or path.name == ".gitkeep":
            continue
        frame = pd.read_csv(path)
        frame["__file__"] = path.stem
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _is_price_folder(frame: pd.DataFrame) -> bool:
    required = {"date", "open", "high", "low", "close", "volume"}
    return required.issubset(frame.columns)


def _write_summary_markdown(title: str, lines: list[str], output_path: Path) -> Path:
    output_path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="utf-8")
    return output_path


def _heatmap_annot(symbol_count: int) -> bool:
    return symbol_count <= 20


def _plot_price_folder(folder_name: str, frame: pd.DataFrame, folder_output: Path) -> dict[str, Path]:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["symbol_key"] = result["symbol"] if "symbol" in result.columns else result["__file__"]
    symbol_count = result["symbol_key"].nunique()

    summary = (
        result.groupby("symbol_key")
        .agg(
            rows=("date", "size"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            close_first=("close", "first"),
            close_last=("close", "last"),
            volume_median=("volume", "median"),
        )
        .reset_index()
        .sort_values("symbol_key")
    )
    summary["close_change_pct"] = summary["close_last"] / summary["close_first"] - 1.0

    pivot_close = result.pivot(index="date", columns="symbol_key", values="close").sort_index()
    rebased = pivot_close.divide(pivot_close.ffill().bfill().iloc[0]).mul(100)

    fig, ax = plt.subplots(figsize=(16, 8 if symbol_count <= 30 else 9))
    rebased.plot(ax=ax, linewidth=1.6)
    ax.set_title(f"{folder_name} Rebased Close (Base = 100, {symbol_count} series)")
    ax.set_xlabel("")
    ax.set_ylabel("Indexed Close")
    if symbol_count <= 20:
        ax.legend(loc="upper left", ncol=2, frameon=True)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    rebased_path = folder_output / "rebased_close.png"
    fig.savefig(rebased_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result["log_volume"] = np.log10(result["volume"].clip(lower=1).astype(float))
    fig, ax = plt.subplots(figsize=(max(12, min(32, symbol_count * 0.35)), 6.5))
    sns.boxplot(data=result, x="symbol_key", y="log_volume", ax=ax)
    ax.set_title(f"{folder_name} Volume Distribution (log10)")
    ax.set_xlabel("")
    ax.set_ylabel("log10(volume)")
    if symbol_count > 20:
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    volume_path = folder_output / "volume_boxplot.png"
    fig.savefig(volume_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result["ret_1d"] = result.groupby("symbol_key")["close"].pct_change()
    corr = result.pivot(index="date", columns="symbol_key", values="ret_1d").corr()
    heatmap_side = max(10, min(28, symbol_count * 0.28))
    fig, ax = plt.subplots(figsize=(heatmap_side, heatmap_side))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=_heatmap_annot(symbol_count),
        fmt=".2f",
        ax=ax,
    )
    ax.set_title(f"{folder_name} Return Correlation")
    if symbol_count > 20:
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    corr_path = folder_output / "return_correlation.png"
    fig.savefig(corr_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_csv = save_table(summary, folder_output / "summary.csv")
    summary_md = _write_summary_markdown(
        title=f"{folder_name} Summary",
        lines=[
            f"- rows: {len(result)}",
            f"- series: {result['symbol_key'].nunique()}",
            f"- date range: {result['date'].min().date().isoformat()} to {result['date'].max().date().isoformat()}",
            (
                "- symbols by source: "
                + ", ".join(
                    f"{source}={count}"
                    for source, count in result.groupby("source")["symbol_key"].nunique().sort_index().items()
                )
                if "source" in result.columns
                else "- symbols by source: unavailable"
            ),
            "",
            "```csv",
            summary.to_csv(index=False).strip(),
            "```",
        ],
        output_path=folder_output / "summary.md",
    )
    return {
        "summary_csv": summary_csv,
        "rebased_close": rebased_path,
        "volume_boxplot": volume_path,
        "return_correlation": corr_path,
        "summary_md": summary_md,
    }


def _plot_reference_folder(folder_name: str, folder: Path, folder_output: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    trading_calendar_path = folder / "trading_calendar.csv"
    if trading_calendar_path.exists():
        calendar = pd.read_csv(trading_calendar_path)
        calendar["date"] = pd.to_datetime(calendar["date"])
        calendar["year_month"] = calendar["date"].dt.to_period("M").astype(str)
        monthly = calendar.groupby("year_month")["date"].size().reset_index(name="sessions")
        fig, ax = plt.subplots(figsize=(16, 5))
        sns.barplot(data=monthly, x="year_month", y="sessions", color="#2a9d8f", ax=ax)
        ax.set_title("Trading Sessions Per Month")
        ax.set_xlabel("")
        ax.set_ylabel("Sessions")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        output_path = folder_output / "trading_sessions_per_month.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs["trading_sessions_per_month"] = output_path

    exchange_mapping_path = folder / "exchange_mapping.csv"
    if exchange_mapping_path.exists():
        exchange = pd.read_csv(exchange_mapping_path)
        if not exchange.empty and {"exchange", "symbol"}.issubset(exchange.columns):
            counts = exchange.groupby("exchange")["symbol"].nunique().reset_index(name="symbols")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.barplot(data=counts, x="exchange", y="symbols", hue="exchange", palette="Set2", legend=False, ax=ax)
            ax.set_title("Symbols by Exchange")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            fig.tight_layout()
            output_path = folder_output / "symbols_by_exchange.png"
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs["symbols_by_exchange"] = output_path

    industry_mapping_path = folder / "industry_mapping.csv"
    if industry_mapping_path.exists():
        industry = pd.read_csv(industry_mapping_path)
        code_column = None
        title = None
        output_name = None
        if not industry.empty and {"symbol", "icb_code2"}.issubset(industry.columns):
            code_column = "icb_code2"
            title = "Symbols by ICB Level-2 Code"
            output_name = "symbols_by_icb_code2.png"
        elif not industry.empty and {"symbol", "industry_code"}.issubset(industry.columns):
            code_column = "industry_code"
            title = "Symbols by Industry Code"
            output_name = "symbols_by_industry_code.png"

        if code_column:
            counts = industry.groupby(code_column)["symbol"].nunique().reset_index(name="symbols")
            counts[code_column] = counts[code_column].astype(str)
            fig, ax = plt.subplots(figsize=(max(8, min(20, len(counts) * 0.55)), 5))
            sns.barplot(
                data=counts,
                x=code_column,
                y="symbols",
                hue=code_column,
                palette="crest",
                legend=False,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel(code_column)
            ax.set_ylabel("Count")
            if len(counts) > 12:
                ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            output_path = folder_output / output_name
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs[output_name.rsplit(".", 1)[0]] = output_path

    foreign_metadata_path = folder / "foreign_symbol_metadata.csv"
    if foreign_metadata_path.exists():
        foreign = pd.read_csv(foreign_metadata_path)
        if not foreign.empty and {"market", "symbol"}.issubset(foreign.columns):
            counts = foreign.groupby("market")["symbol"].nunique().reset_index(name="symbols")
            fig, ax = plt.subplots(figsize=(max(7, len(counts) * 1.3), 5))
            sns.barplot(
                data=counts,
                x="market",
                y="symbols",
                hue="market",
                palette="rocket",
                legend=False,
                ax=ax,
            )
            ax.set_title("Foreign Symbols by Market")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            fig.tight_layout()
            output_path = folder_output / "foreign_symbols_by_market.png"
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs["foreign_symbols_by_market"] = output_path

    json_paths = [path for path in _list_data_files(folder) if path.suffix == ".json"]
    manifest_preview: dict[str, Any] = {}
    for path in json_paths:
        try:
            manifest_preview[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest_preview[path.name] = {"error": "invalid json"}

    summary_md = _write_summary_markdown(
        title=f"{folder_name} Summary",
        lines=[
            f"- files: {len(_list_data_files(folder))}",
            f"- generated charts: {len(outputs)}",
            "",
            "```json",
            json.dumps(manifest_preview, ensure_ascii=False, indent=2)[:6000],
            "```",
        ],
        output_path=folder_output / "summary.md",
    )
    outputs["summary_md"] = summary_md
    return outputs


def visualize_raw_data_folders(base_dir: Path | None = None) -> dict[str, dict[str, str]]:
    raw_root = base_dir or _raw_data_root()
    figures_root = _figures_root()
    outputs: dict[str, dict[str, str]] = {}

    for folder in sorted([path for path in raw_root.iterdir() if path.is_dir() and not path.name.startswith("_")]):
        folder_output = ensure_dir(figures_root / folder.name)
        frame = _load_csv_folder(folder)

        if not frame.empty and _is_price_folder(frame):
            folder_outputs = _plot_price_folder(folder.name, frame, folder_output)
        else:
            folder_outputs = _plot_reference_folder(folder.name, folder, folder_output)

        outputs[folder.name] = {name: str(path) for name, path in folder_outputs.items()}

    return outputs
