from __future__ import annotations

import pandas as pd

from vnstock.data.cleaning import standardize_raw_panel
from vnstock.utils.schema import RAW_COLUMNS


def test_standardize_raw_panel_enforces_unique_symbol_date() -> None:
    frame = pd.DataFrame(
        [
            {"symbol": "vcb", "date": "2024-01-02", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100, "value": 1050, "source": "KBS"},
            {"symbol": "VCB", "date": "2024-01-02", "open": 10, "high": 11, "low": 9, "close": 10.6, "volume": 110, "value": 1166, "source": "KBS"},
        ]
    )

    cleaned = standardize_raw_panel(frame)
    assert list(cleaned.columns) == RAW_COLUMNS
    assert cleaned.shape[0] == 1
    assert cleaned.loc[0, "symbol"] == "VCB"

