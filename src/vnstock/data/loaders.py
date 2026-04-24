from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnstock.data.cleaning import standardize_raw_panel
from vnstock.data.validation import validate_required_columns
from vnstock.utils.io import load_table
from vnstock.utils.schema import RAW_COLUMNS


def load_universe(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip().upper() for line in handle if line.strip()]


def load_raw_symbol_file(path: str | Path) -> pd.DataFrame:
    frame = load_table(path)
    validate_required_columns(frame, RAW_COLUMNS)
    return standardize_raw_panel(frame)


def load_raw_panel(raw_root: str | Path, universe: list[str] | None = None) -> pd.DataFrame:
    root = Path(raw_root)
    csv_files = sorted(root.glob("*.csv"))
    if universe:
        csv_files = [path for path in csv_files if path.stem.upper() in set(universe)]

    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found under {root}")

    frames = [load_raw_symbol_file(path) for path in csv_files]
    return standardize_raw_panel(pd.concat(frames, ignore_index=True))


def load_prediction_files(predictions_root: str | Path) -> list[Path]:
    return sorted(Path(predictions_root).glob("*_predictions.parquet"))

