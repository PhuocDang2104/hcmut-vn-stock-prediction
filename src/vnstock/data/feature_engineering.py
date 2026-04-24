from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "log_volume",
    "hl_spread",
    "oc_change",
    "rolling_vol_5",
    "rolling_vol_20",
    "ma_ratio_5_20",
    "open_rel",
    "high_rel",
    "low_rel",
    "ma5_rel",
    "ma20_rel",
]

KRONOS_FEATURE_COLUMNS = [
    "open_rel",
    "high_rel",
    "low_rel",
    "close_ret",
    "log_volume",
]


def add_common_features(frame: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    result = frame.sort_values(["symbol", "date"]).copy()
    grouped = result.groupby("symbol", group_keys=False)

    result["ret_1d"] = grouped["close"].pct_change(1)
    result["ret_5d"] = grouped["close"].pct_change(5)
    result["close_ret"] = result["ret_1d"]
    result["log_volume"] = np.log1p(result["volume"].clip(lower=0))
    result["hl_spread"] = (result["high"] - result["low"]) / result["close"].replace(0, np.nan)
    result["oc_change"] = (result["close"] - result["open"]) / result["open"].replace(0, np.nan)
    result["rolling_vol_5"] = grouped["ret_1d"].transform(lambda s: s.rolling(5).std())
    result["rolling_vol_20"] = grouped["ret_1d"].transform(lambda s: s.rolling(20).std())
    result["ma_5"] = grouped["close"].transform(lambda s: s.rolling(5).mean())
    result["ma_20"] = grouped["close"].transform(lambda s: s.rolling(20).mean())
    result["ma_ratio_5_20"] = result["ma_5"] / result["ma_20"].replace(0, np.nan)
    result["open_rel"] = result["open"] / result["close"].replace(0, np.nan) - 1.0
    result["high_rel"] = result["high"] / result["close"].replace(0, np.nan) - 1.0
    result["low_rel"] = result["low"] / result["close"].replace(0, np.nan) - 1.0
    result["ma5_rel"] = result["ma_5"] / result["close"].replace(0, np.nan) - 1.0
    result["ma20_rel"] = result["ma_20"] / result["close"].replace(0, np.nan) - 1.0
    result["target_ret_1d"] = grouped["close"].shift(-1) / result["close"] - 1.0
    result["target_ret_5d"] = grouped["close"].shift(-horizon) / result["close"] - 1.0
    result["target_dir_5d"] = (result["target_ret_5d"] > 0).astype("Int64")
    result["time_idx"] = grouped.cumcount()
    result["group_id"] = result["symbol"]
    return result


def feature_columns_from_config(config: dict) -> list[str]:
    features = config.get("features")
    if not features:
        return DEFAULT_FEATURE_COLUMNS.copy()
    return list(features)


def fill_feature_nans(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result[list(feature_columns)] = result.groupby("symbol")[list(feature_columns)].transform(
        lambda s: s.ffill()
    )
    return result
