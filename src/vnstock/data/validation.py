from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from vnstock.utils.schema import PREDICTION_COLUMNS


def validate_required_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_unique_rows(frame: pd.DataFrame, subset: Sequence[str]) -> None:
    duplicate_mask = frame.duplicated(subset=list(subset), keep=False)
    if duplicate_mask.any():
        duplicates = frame.loc[duplicate_mask, list(subset)].head().to_dict(orient="records")
        raise ValueError(f"Duplicate rows found for {list(subset)}: {duplicates}")


def ensure_datetime_column(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise")
    return result


def validate_monotonic_dates(frame: pd.DataFrame) -> None:
    ordered = frame.sort_values(["symbol", "date"])
    monotonic = ordered.groupby("symbol")["date"].apply(lambda s: s.is_monotonic_increasing).all()
    if not monotonic:
        raise ValueError("Dates are not monotonically increasing within every symbol.")


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    validate_required_columns(frame, PREDICTION_COLUMNS)
    validate_unique_rows(frame, ["model_family", "symbol", "date", "run_id"])

