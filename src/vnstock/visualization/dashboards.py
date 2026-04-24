from __future__ import annotations

import pandas as pd


def consensus_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["date", "symbol"])
        .agg(
            y_true=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
            pred_std=("y_pred", "std"),
            models=("model_family", "nunique"),
        )
        .reset_index()
    )

