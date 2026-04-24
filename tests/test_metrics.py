from __future__ import annotations

import pandas as pd

from vnstock.evaluation.metrics import compute_metrics


def test_compute_metrics_returns_expected_keys() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "y_true": [0.02, -0.01, 0.03, -0.02],
            "y_pred": [0.01, -0.02, 0.02, -0.01],
        }
    )

    metrics = compute_metrics(frame)
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert 0 <= metrics["directional_accuracy"] <= 1
    assert "information_coefficient" in metrics

