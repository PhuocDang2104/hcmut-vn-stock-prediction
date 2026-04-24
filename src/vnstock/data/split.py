from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: str
    valid_end: str
    test_end: str


def assign_time_split(frame: pd.DataFrame, split_config: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    date_index = pd.to_datetime(result["date"])
    train_end = pd.Timestamp(split_config["train_end"])
    valid_end = pd.Timestamp(split_config["valid_end"])
    test_end = pd.Timestamp(split_config["test_end"])

    conditions = [
        date_index <= train_end,
        (date_index > train_end) & (date_index <= valid_end),
        (date_index > valid_end) & (date_index <= test_end),
    ]
    choices = ["train", "valid", "test"]
    result["split"] = np.select(conditions, choices, default="discard")
    return result


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": frame.loc[frame["split"] == "train"].copy(),
        "valid": frame.loc[frame["split"] == "valid"].copy(),
        "test": frame.loc[frame["split"] == "test"].copy(),
    }

