from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnstock.data.loaders import load_prediction_files
from vnstock.evaluation.metrics import mae
from vnstock.utils.io import load_table, save_table


def load_all_predictions(predictions_dir: str | Path) -> pd.DataFrame:
    files = load_prediction_files(predictions_dir)
    if not files:
        return pd.DataFrame()
    frames = [load_table(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def compare_by_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["model_family", "symbol"])
        .apply(lambda group: pd.Series({"mae": mae(group["y_true"].to_numpy(), group["y_pred"].to_numpy()), "rows": len(group)}))
        .reset_index()
    )


def export_all_predictions(predictions_dir: str | Path, output_path: str | Path) -> Path:
    combined = load_all_predictions(predictions_dir)
    return save_table(combined, output_path)

