from __future__ import annotations

import pandas as pd

from vnstock.data.fetch import _normalize_yfinance_frame, resolve_symbol_source
from vnstock.utils.schema import RAW_COLUMNS


def test_resolve_symbol_source_uses_override() -> None:
    config = {
        "source": "KBS",
        "symbol_source_overrides": {
            "AAPL": "YF",
        },
    }

    assert resolve_symbol_source(config, "AAPL") == "YF"
    assert resolve_symbol_source(config, "VCB") == "KBS"


def test_normalize_yfinance_frame_matches_raw_schema() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame(
        {
            ("Open", "AAPL"): [100.0, 101.0],
            ("High", "AAPL"): [102.0, 103.0],
            ("Low", "AAPL"): [99.0, 100.0],
            ("Close", "AAPL"): [101.0, 102.0],
            ("Volume", "AAPL"): [1000, 1100],
        },
        index=index,
    )
    frame.index.name = "Date"

    normalized = _normalize_yfinance_frame(
        frame=frame,
        symbol="AAPL",
        source="YF",
        start_date="2024-01-01",
        end_date="2024-01-31",
        value_mode=None,
    )

    assert list(normalized.columns) == RAW_COLUMNS
    assert normalized["symbol"].tolist() == ["AAPL", "AAPL"]
    assert normalized["source"].tolist() == ["YF", "YF"]
    assert normalized["date"].tolist() == ["2024-01-02", "2024-01-03"]
