from __future__ import annotations

import pandas as pd

from vnstock.models.common.base_predictor import standardize_prediction_output
from vnstock.utils.schema import PREDICTION_COLUMNS, PredictionContext


def test_standardize_prediction_output_matches_contract() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["VCB", "TCB"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "split": ["test", "test"],
            "target_ret_5d": [0.01, -0.02],
            "y_pred": [0.02, -0.01],
        }
    )
    context = PredictionContext(
        model_family="xlstm_ts",
        model_version="xlstm_ts_v1",
        target_name="target_ret_5d",
        horizon=5,
        run_id="run_test",
    )

    output = standardize_prediction_output(frame, context)
    assert output.columns.tolist() == PREDICTION_COLUMNS
    assert output["model_family"].unique().tolist() == ["xlstm_ts"]
