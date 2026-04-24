from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnstock.data.cleaning import standardize_raw_panel
from vnstock.data.loaders import load_raw_panel, load_universe
from vnstock.data.validation import validate_monotonic_dates
from vnstock.utils.io import save_table, write_json
from vnstock.utils.paths import path_for, resolve_path


def build_quality_report(frame: pd.DataFrame) -> dict[str, object]:
    date_series = pd.to_datetime(frame["date"])
    return {
        "symbols": int(frame["symbol"].nunique()),
        "rows": int(len(frame)),
        "date_min": date_series.min().date().isoformat(),
        "date_max": date_series.max().date().isoformat(),
        "missing_close": int(frame["close"].isna().sum()),
        "duplicates": int(frame.duplicated(["symbol", "date"]).sum()),
    }


def ingest_repo_csvs(config: dict) -> dict[str, Path]:
    raw_root = path_for("raw_root")
    interim_root = path_for("interim_root")
    broad_universe_path = resolve_path(config["universes"]["broad"])
    universe = load_universe(broad_universe_path)

    raw_panel = load_raw_panel(raw_root, universe=universe)
    raw_panel = standardize_raw_panel(raw_panel)
    validate_monotonic_dates(raw_panel)

    merged_path = save_table(raw_panel, interim_root / "merged_panel.parquet")
    cleaned_path = save_table(raw_panel, interim_root / "cleaned_panel.parquet")
    report_path = write_json(build_quality_report(raw_panel), interim_root / "quality_report.json")
    return {
        "merged_panel": merged_path,
        "cleaned_panel": cleaned_path,
        "quality_report": report_path,
    }

