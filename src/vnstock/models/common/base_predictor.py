from __future__ import annotations

import numpy as np
import pandas as pd

from vnstock.data.validation import validate_prediction_frame
from vnstock.utils.schema import PREDICTION_COLUMNS, PredictionContext


def standardize_prediction_output(
    frame: pd.DataFrame,
    context: PredictionContext,
    y_true_column: str = "target_ret_5d",
    y_pred_column: str = "y_pred",
) -> pd.DataFrame:
    required = {"symbol", "date", "split", y_true_column}
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction export: {missing}")

    y_pred_values = frame[y_pred_column] if y_pred_column in frame.columns else np.nan
    output = pd.DataFrame(
        {
            "model_family": context.model_family,
            "model_version": context.model_version,
            "symbol": frame["symbol"].astype(str),
            "date": pd.to_datetime(frame["date"]),
            "split": frame["split"].astype(str),
            "y_true": frame[y_true_column].astype(float),
            "y_pred": pd.Series(y_pred_values, index=frame.index, dtype=float),
            "target_name": context.target_name,
            "horizon": int(context.horizon),
            "run_id": context.run_id,
        }
    )[PREDICTION_COLUMNS]
    validate_prediction_frame(output)
    return output


class BasePredictor:
    def __init__(self, context: PredictionContext) -> None:
        self.context = context

    def export(self, frame: pd.DataFrame, y_true_column: str = "target_ret_5d") -> pd.DataFrame:
        return standardize_prediction_output(frame, self.context, y_true_column=y_true_column)

