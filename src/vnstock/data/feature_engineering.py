from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
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
    result["ret_3d"] = grouped["close"].pct_change(3)
    result["ret_5d"] = grouped["close"].pct_change(5)
    result["ret_10d"] = grouped["close"].pct_change(10)
    result["ret_20d"] = grouped["close"].pct_change(20)
    result["log_close"] = np.log(result["close"].replace(0, np.nan))
    result["close_ret"] = result["ret_1d"]
    result["log_volume"] = np.log1p(result["volume"].clip(lower=0))
    log_volume_grouped = result.groupby("symbol", group_keys=False)["log_volume"]
    volume_mean_20 = log_volume_grouped.transform(lambda s: s.rolling(20).mean())
    volume_std_20 = log_volume_grouped.transform(lambda s: s.rolling(20).std())
    result["volume_zscore_20"] = (result["log_volume"] - volume_mean_20) / volume_std_20.replace(0, np.nan)
    result["hl_spread"] = (result["high"] - result["low"]) / result["close"].replace(0, np.nan)
    result["oc_change"] = (result["close"] - result["open"]) / result["open"].replace(0, np.nan)
    previous_close = grouped["close"].shift(1)
    result["gap_rel"] = result["open"] / previous_close.replace(0, np.nan) - 1.0
    candle_range = (result["high"] - result["low"]).replace(0, np.nan)
    result["body_ratio"] = (result["close"] - result["open"]) / candle_range
    result["upper_shadow_ratio"] = (result["high"] - result[["open", "close"]].max(axis=1)) / candle_range
    result["lower_shadow_ratio"] = (result[["open", "close"]].min(axis=1) - result["low"]) / candle_range
    result["rolling_vol_5"] = grouped["ret_1d"].transform(lambda s: s.rolling(5).std())
    result["rolling_vol_20"] = grouped["ret_1d"].transform(lambda s: s.rolling(20).std())
    result["vol_ratio_5_20"] = result["rolling_vol_5"] / result["rolling_vol_20"].replace(0, np.nan)
    result["ma_5"] = grouped["close"].transform(lambda s: s.rolling(5).mean())
    result["ma_20"] = grouped["close"].transform(lambda s: s.rolling(20).mean())
    result["ma_ratio_5_20"] = result["ma_5"] / result["ma_20"].replace(0, np.nan)
    result["open_rel"] = result["open"] / result["close"].replace(0, np.nan) - 1.0
    result["high_rel"] = result["high"] / result["close"].replace(0, np.nan) - 1.0
    result["low_rel"] = result["low"] / result["close"].replace(0, np.nan) - 1.0
    result["ma5_rel"] = result["ma_5"] / result["close"].replace(0, np.nan) - 1.0
    result["ma20_rel"] = result["ma_20"] / result["close"].replace(0, np.nan) - 1.0
    close_diff = grouped["close"].diff()
    gains = close_diff.clip(lower=0)
    losses = -close_diff.clip(upper=0)
    avg_gain_14 = gains.groupby(result["symbol"]).transform(lambda s: s.rolling(14).mean())
    avg_loss_14 = losses.groupby(result["symbol"]).transform(lambda s: s.rolling(14).mean())
    rs_14 = avg_gain_14 / avg_loss_14.replace(0, np.nan)
    result["rsi_14"] = 100.0 - 100.0 / (1.0 + rs_14)
    ema_12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
    ema_26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
    macd = ema_12 - ema_26
    macd_signal = macd.groupby(result["symbol"]).transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    result["macd_hist"] = (macd - macd_signal) / result["close"].replace(0, np.nan)
    by_date = result.groupby("date", group_keys=False)
    result["market_ret_1d"] = by_date["ret_1d"].transform("mean")
    result["market_ret_5d"] = by_date["ret_5d"].transform("mean")
    result["market_ret_20d"] = by_date["ret_20d"].transform("mean")
    result["market_vol_20"] = by_date["rolling_vol_20"].transform("mean")
    result["excess_ret_1d"] = result["ret_1d"] - result["market_ret_1d"]
    result["excess_ret_5d"] = result["ret_5d"] - result["market_ret_5d"]
    result["excess_ret_20d"] = result["ret_20d"] - result["market_ret_20d"]
    result["relative_strength_5d"] = result["ret_5d"] - result["market_ret_5d"]
    result["relative_strength_20d"] = result["ret_20d"] - result["market_ret_20d"]
    for target_horizon in (1, 3, 5):
        future_close = grouped["close"].shift(-target_horizon)
        result[f"target_ret_{target_horizon}d"] = future_close / result["close"] - 1.0
        result[f"target_log_ret_{target_horizon}d"] = np.log(future_close / result["close"])
        result[f"target_log_close_{target_horizon}d"] = np.log(future_close.replace(0, np.nan))
        result[f"target_dir_{target_horizon}d"] = (result[f"target_ret_{target_horizon}d"] > 0).astype("Int64")
        result[f"target_excess_ret_{target_horizon}d"] = (
            result[f"target_ret_{target_horizon}d"] - by_date[f"target_ret_{target_horizon}d"].transform("mean")
        )
    if horizon not in (1, 3, 5):
        future_close = grouped["close"].shift(-horizon)
        result[f"target_ret_{horizon}d"] = future_close / result["close"] - 1.0
        result[f"target_log_ret_{horizon}d"] = np.log(future_close / result["close"])
        result[f"target_log_close_{horizon}d"] = np.log(future_close.replace(0, np.nan))
        result[f"target_dir_{horizon}d"] = (result[f"target_ret_{horizon}d"] > 0).astype("Int64")
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
