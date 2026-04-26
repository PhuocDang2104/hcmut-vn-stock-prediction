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
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "log_close",
    "log_volume",
    "volume_zscore_20",
    "hl_spread",
    "oc_change",
    "gap_rel",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "rolling_vol_5",
    "rolling_vol_20",
    "vol_ratio_5_20",
    "close_ret",
    "ma_5",
    "ma_20",
    "ma_ratio_5_20",
    "open_rel",
    "high_rel",
    "low_rel",
    "ma5_rel",
    "ma20_rel",
    "rsi_14",
    "macd_hist",
    "market_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_vol_20",
    "excess_ret_1d",
    "excess_ret_5d",
    "excess_ret_20d",
    "relative_strength_5d",
    "relative_strength_20d",
    "target_ret_1d",
    "target_ret_3d",
    "target_ret_5d",
    "target_log_ret_1d",
    "target_log_ret_3d",
    "target_log_ret_5d",
    "target_log_close_1d",
    "target_log_close_3d",
    "target_log_close_5d",
    "target_excess_ret_1d",
    "target_excess_ret_3d",
    "target_excess_ret_5d",
    "target_dir_1d",
    "target_dir_3d",
    "target_dir_5d",
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
