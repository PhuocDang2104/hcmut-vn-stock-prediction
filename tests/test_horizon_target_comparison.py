from __future__ import annotations

import numpy as np
import pandas as pd

from vnstock.data.feature_engineering import add_common_features
from vnstock.models.common.sequence_data import build_scaled_sequence_splits
from vnstock.pipelines.run_horizon_target_comparison import (
    daily_correlation,
    tune_threshold,
)


def test_target_generation_horizons_1_3_5() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 10,
            "date": pd.date_range("2024-01-01", periods=10),
            "open": np.arange(10, 20, dtype=float),
            "high": np.arange(11, 21, dtype=float),
            "low": np.arange(9, 19, dtype=float),
            "close": np.arange(10, 20, dtype=float),
            "volume": np.arange(100, 110, dtype=float),
            "value": np.arange(1000, 1010, dtype=float),
            "source": ["test"] * 10,
        }
    )

    result = add_common_features(frame)

    assert np.isclose(result.loc[0, "target_ret_1d"], 11 / 10 - 1)
    assert np.isclose(result.loc[0, "target_ret_3d"], 13 / 10 - 1)
    assert np.isclose(result.loc[0, "target_ret_5d"], 15 / 10 - 1)
    assert np.isclose(result.loc[0, "target_log_close_3d"], np.log(13))
    assert result.loc[0, "target_dir_5d"] == 1


def test_target_scaler_fit_train_only() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["AAA"] * 8,
            "date": pd.date_range("2024-01-01", periods=8),
            "split": ["train", "train", "train", "train", "valid", "valid", "test", "test"],
            "feature": np.arange(8, dtype=float),
            "target": [10.0, 12.0, 14.0, 16.0, 1000.0, 1200.0, 1400.0, 1600.0],
        }
    )

    splits, scalers = build_scaled_sequence_splits(
        feature_panel=panel,
        feature_columns=["feature"],
        target_column="target",
        lookback=2,
        target_scaling={"enabled": True},
    )

    train_mean = np.mean([10.0, 12.0, 14.0, 16.0])
    assert np.isclose(scalers["__target__"]["AAA"]["mean"], train_mean)
    assert "target_raw" in splits["valid"].meta.columns
    assert np.isclose(splits["valid"].meta.iloc[0]["target_mean"], train_mean)


def test_wavelet_is_applied_per_historical_window_only() -> None:
    base = pd.DataFrame(
        {
            "symbol": ["AAA"] * 8,
            "date": pd.date_range("2024-01-01", periods=8),
            "split": ["train", "train", "train", "train", "valid", "valid", "test", "test"],
            "feature": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1000.0, 2000.0],
            "target": np.arange(8, dtype=float),
        }
    )
    changed_future = base.copy()
    changed_future.loc[changed_future["split"] == "test", "feature"] = [999999.0, 888888.0]
    kwargs = {
        "feature_columns": ["feature"],
        "target_column": "target",
        "lookback": 4,
        "wavelet_config": {
            "enabled": True,
            "append": True,
            "level": 1,
            "wavelet": "db4",
            "feature_columns": ["feature"],
        },
    }

    base_splits, _ = build_scaled_sequence_splits(feature_panel=base, **kwargs)
    changed_splits, _ = build_scaled_sequence_splits(feature_panel=changed_future, **kwargs)

    np.testing.assert_allclose(base_splits["valid"].X, changed_splits["valid"].X)


def test_daily_ic_and_rankic_are_cross_sectional() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "ranking_score": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "true_ret": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )

    ic, _ = daily_correlation(frame, method="pearson")
    rankic, _ = daily_correlation(frame, method="spearman")

    assert np.isclose(ic, 0.0)
    assert np.isclose(rankic, 0.0)


def test_threshold_tuning_uses_supplied_validation_frame_only() -> None:
    valid = pd.DataFrame(
        {
            "true_ret": [-0.02, -0.01, 0.01, 0.02],
            "pred_ret": [-0.02, -0.01, 0.01, 0.02],
        }
    )
    test_like = pd.DataFrame(
        {
            "true_ret": [0.02, 0.01, -0.01, -0.02],
            "pred_ret": [-0.02, -0.01, 0.01, 0.02],
        }
    )

    threshold_valid = tune_threshold(valid, score_column="pred_ret", thresholds=[-0.01, 0.0, 0.01])
    threshold_test_like = tune_threshold(test_like, score_column="pred_ret", thresholds=[-0.01, 0.0, 0.01])

    assert threshold_valid != threshold_test_like
