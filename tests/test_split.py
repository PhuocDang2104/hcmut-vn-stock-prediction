from __future__ import annotations

import pandas as pd

from vnstock.data.split import assign_time_split


def test_assign_time_split_uses_time_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["VCB", "VCB", "VCB", "VCB"],
            "date": pd.to_datetime(["2023-12-31", "2024-06-30", "2025-01-02", "2027-01-01"]),
        }
    )
    split_config = {
        "train_end": "2023-12-31",
        "valid_end": "2024-12-31",
        "test_end": "2026-12-31",
    }

    result = assign_time_split(frame, split_config)
    assert result["split"].tolist() == ["train", "valid", "test", "discard"]

