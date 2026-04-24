from __future__ import annotations

from dataclasses import dataclass


RAW_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "source",
]

SHARED_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "ret_1d",
    "ret_5d",
    "log_volume",
    "hl_spread",
    "oc_change",
    "rolling_vol_5",
    "rolling_vol_20",
    "close_ret",
    "ma_5",
    "ma_20",
    "ma_ratio_5_20",
    "open_rel",
    "high_rel",
    "low_rel",
    "ma5_rel",
    "ma20_rel",
    "target_ret_1d",
    "target_ret_5d",
    "split",
]

PREDICTION_COLUMNS = [
    "model_family",
    "model_version",
    "symbol",
    "date",
    "split",
    "y_true",
    "y_pred",
    "target_name",
    "horizon",
    "run_id",
]


@dataclass(frozen=True)
class PredictionContext:
    model_family: str
    model_version: str
    target_name: str
    horizon: int
    run_id: str
