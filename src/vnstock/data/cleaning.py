from __future__ import annotations

import pandas as pd

from vnstock.data.validation import ensure_datetime_column, validate_required_columns, validate_unique_rows
from vnstock.utils.schema import RAW_COLUMNS


NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "value"]


def standardize_raw_panel(frame: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(frame, RAW_COLUMNS)

    result = ensure_datetime_column(frame, "date").copy()
    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    result["source"] = result["source"].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["symbol", "date", "close"])
    result = result.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    validate_unique_rows(result, ["symbol", "date"])
    return result.reset_index(drop=True)

