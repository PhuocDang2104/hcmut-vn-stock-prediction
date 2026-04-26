from __future__ import annotations

import pandas as pd

from vnstock.models.common.base_predictor import standardize_prediction_output
from vnstock.utils.schema import PredictionContext


def build_prediction_frame(frame: pd.DataFrame, context: PredictionContext) -> pd.DataFrame:
    return standardize_prediction_output(frame, context, y_true_column=context.target_name)
