from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnstock.data.loaders import load_prediction_files
from vnstock.evaluation.backtest_proxy import compute_backtest_proxy
from vnstock.evaluation.metrics import compute_metrics
from vnstock.utils.io import load_table, save_table


def build_leaderboard(
    predictions_dir: str | Path,
    split: str | None = "test",
    align_intersection: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    loaded_frames: list[tuple[Path, pd.DataFrame]] = []
    for path in load_prediction_files(predictions_dir):
        frame = load_table(path)
        if split and "split" in frame.columns and split in set(frame["split"].astype(str)):
            frame = frame.loc[frame["split"].astype(str) == split].copy()
        loaded_frames.append((path, frame))

    if align_intersection and loaded_frames:
        common_keys = None
        for _, frame in loaded_frames:
            keys = set(zip(frame["symbol"].astype(str), pd.to_datetime(frame["date"])))
            common_keys = keys if common_keys is None else common_keys & keys
        common_keys = common_keys or set()
        for index, (path, frame) in enumerate(loaded_frames):
            key_index = list(zip(frame["symbol"].astype(str), pd.to_datetime(frame["date"])))
            mask = [key in common_keys for key in key_index]
            loaded_frames[index] = (path, frame.loc[mask].copy())

    for path, frame in loaded_frames:
        model_name = str(frame["model_family"].iloc[0]) if not frame.empty else path.stem.replace("_predictions", "")
        metrics = compute_metrics(frame)
        proxy = compute_backtest_proxy(frame)
        records.append(
            {
                "model_family": model_name,
                "evaluated_split": split or "all",
                "rows": int(len(frame)),
                **metrics,
                **proxy,
                "source_file": str(path),
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["huber", "rmse", "mae"], ascending=[True, True, True])


def export_leaderboard(
    predictions_dir: str | Path,
    output_path: str | Path,
    split: str | None = "test",
    align_intersection: bool = False,
) -> Path:
    leaderboard = build_leaderboard(
        predictions_dir,
        split=split,
        align_intersection=align_intersection,
    )
    return save_table(leaderboard, output_path)
