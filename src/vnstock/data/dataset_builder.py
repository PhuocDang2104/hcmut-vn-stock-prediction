from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vnstock.data.feature_engineering import (
    DEFAULT_FEATURE_COLUMNS,
    KRONOS_FEATURE_COLUMNS,
    add_common_features,
    fill_feature_nans,
)
from vnstock.data.split import assign_time_split, split_frame
from vnstock.data.validation import validate_monotonic_dates
from vnstock.utils.io import ensure_dir, save_table, write_json
from vnstock.utils.paths import path_for
from vnstock.utils.schema import SHARED_COLUMNS


@dataclass
class SharedDatasetBundle:
    feature_panel: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    metadata: dict[str, Any]


def _shared_root() -> Path:
    return path_for("shared_root")


def _processed_root() -> Path:
    return path_for("processed_root")


def build_shared_feature_panel(raw_panel: pd.DataFrame, config: dict) -> SharedDatasetBundle:
    featured = add_common_features(raw_panel, horizon=int(config["target"]["horizon"]))
    featured = assign_time_split(featured, config["split"])
    featured = featured.loc[featured["split"] != "discard"].copy()
    feature_columns = list(config.get("features", DEFAULT_FEATURE_COLUMNS))
    featured = fill_feature_nans(featured, feature_columns)
    featured = featured.dropna(subset=feature_columns + ["target_ret_5d"]).reset_index(drop=True)
    validate_monotonic_dates(featured)

    feature_panel = featured[SHARED_COLUMNS + ["time_idx", "group_id", "target_dir_5d"]].copy()
    splits = split_frame(feature_panel)
    metadata = {
        "target_name": config["target"]["name"],
        "horizon": int(config["target"]["horizon"]),
        "feature_columns": feature_columns,
        "kronos_feature_columns": list(
            config.get("kronos_features", config.get("model_exports", {}).get("kronos", {}).get("feature_columns", KRONOS_FEATURE_COLUMNS))
        ),
        "split": config["split"],
    }
    return SharedDatasetBundle(feature_panel=feature_panel, splits=splits, metadata=metadata)


def export_shared_bundle(bundle: SharedDatasetBundle) -> dict[str, Path]:
    shared_root = _shared_root()
    ensure_dir(shared_root)

    output_paths = {
        "market_panel": save_table(bundle.feature_panel, shared_root / "market_panel.parquet"),
        "feature_panel": save_table(bundle.feature_panel, shared_root / "feature_panel.parquet"),
        "train": save_table(bundle.splits["train"], shared_root / "train.parquet"),
        "valid": save_table(bundle.splits["valid"], shared_root / "valid.parquet"),
        "test": save_table(bundle.splits["test"], shared_root / "test.parquet"),
        "schema": write_json({"columns": list(bundle.feature_panel.columns)}, shared_root / "schema.json"),
        "split_meta": write_json(bundle.metadata, shared_root / "split_meta.json"),
    }
    return output_paths


def build_sequence_arrays(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_name: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []

    for _, group in feature_panel.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_columns].to_numpy(dtype=float)
        target_values = group[target_column].to_numpy(dtype=float)
        split_values = group["split"].to_numpy(dtype=object)

        for end_idx in range(lookback - 1, len(group)):
            if split_values[end_idx] != split_name:
                continue
            window = values[end_idx - lookback + 1 : end_idx + 1]
            if np.isnan(window).any() or np.isnan(target_values[end_idx]):
                continue
            sequences.append(window)
            targets.append(float(target_values[end_idx]))

    if not sequences:
        return np.empty((0, lookback, len(feature_columns))), np.empty((0,))

    return np.stack(sequences), np.asarray(targets, dtype=float)


def export_array_model_dataset(
    feature_panel: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
    window_field_name: str = "lookback",
) -> dict[str, Path]:
    output_dir = ensure_dir(_processed_root() / model_name)
    summary: dict[str, Any] = {
        "model_name": model_name,
        window_field_name: lookback,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }

    exported: dict[str, Path] = {}
    for split_name in ("train", "valid", "test"):
        x_values, y_values = build_sequence_arrays(
            feature_panel=feature_panel,
            feature_columns=feature_columns,
            target_column=target_column,
            split_name=split_name,
            lookback=lookback,
        )
        x_path = output_dir / f"X_{split_name}.npy"
        y_path = output_dir / f"y_{split_name}.npy"
        np.save(x_path, x_values)
        np.save(y_path, y_values)
        exported[f"X_{split_name}"] = x_path
        exported[f"y_{split_name}"] = y_path
        summary[f"{split_name}_samples"] = int(len(y_values))

    exported["meta"] = write_json(summary, output_dir / "meta.json")
    return exported


def export_panel_model_dataset(
    feature_panel: pd.DataFrame,
    model_name: str,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = ensure_dir(_processed_root() / model_name)
    splits = split_frame(feature_panel)
    output_paths = {
        split_name: save_table(split_frame_data, output_dir / f"{split_name}.parquet")
        for split_name, split_frame_data in splits.items()
    }
    metadata = {
        "model_name": model_name,
        "columns": list(feature_panel.columns),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    output_paths["meta"] = write_json(
        metadata,
        output_dir / "meta.json",
    )
    return output_paths


def export_xlstm_ts_dataset(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    context_length: int,
) -> dict[str, Path]:
    return export_array_model_dataset(
        feature_panel=feature_panel,
        model_name="xlstm_ts",
        feature_columns=feature_columns,
        target_column=target_column,
        lookback=context_length,
        window_field_name="context_length",
    )


def export_patchtst_dataset(feature_panel: pd.DataFrame, config: dict[str, Any]) -> dict[str, Path]:
    patchtst_config = config["model_exports"]["patchtst"]
    return export_panel_model_dataset(
        feature_panel,
        "patchtst",
        metadata_extra={
            "seq_len": int(patchtst_config["seq_len"]),
            "pred_len": int(patchtst_config["pred_len"]),
            "patch_len": int(patchtst_config["patch_len"]),
            "patch_stride": int(patchtst_config["patch_stride"]),
        },
    )


def export_kronos_dataset(feature_panel: pd.DataFrame) -> dict[str, Path]:
    output_dir = ensure_dir(_processed_root() / "kronos")
    kronos_frame = feature_panel.copy()
    kronos_frame["timestamp"] = pd.to_datetime(kronos_frame["date"]).dt.strftime("%Y-%m-%d")
    kronos_frame["target"] = kronos_frame["target_ret_5d"]
    splits = split_frame(kronos_frame)

    output_paths = {
        split_name: save_table(split_frame_data, output_dir / f"{split_name}.csv")
        for split_name, split_frame_data in splits.items()
    }
    output_paths["meta"] = write_json(
        {
            "model_name": "kronos",
            "adapter_mode": "vn_equity_daily",
            "columns": list(kronos_frame.columns),
            "feature_columns": KRONOS_FEATURE_COLUMNS,
        },
        output_dir / "meta.json",
    )
    return output_paths


def build_all_datasets(raw_panel: pd.DataFrame, config: dict) -> dict[str, dict[str, Path]]:
    bundle = build_shared_feature_panel(raw_panel, config)
    feature_columns = list(bundle.metadata["feature_columns"])
    target_column = config["target"]["name"]

    exports = {
        "shared": export_shared_bundle(bundle),
        "xlstm_ts": export_xlstm_ts_dataset(
            bundle.feature_panel,
            feature_columns=feature_columns,
            target_column=target_column,
            context_length=int(config["model_exports"]["xlstm_ts"]["context_length"]),
        ),
        "itransformer": export_panel_model_dataset(
            bundle.feature_panel,
            "itransformer",
            metadata_extra={
                "seq_len": int(config["model_exports"]["itransformer"]["seq_len"]),
                "pred_len": int(config["model_exports"]["itransformer"]["pred_len"]),
            },
        ),
        "patchtst": export_patchtst_dataset(bundle.feature_panel, config),
        "kronos": export_kronos_dataset(bundle.feature_panel),
    }
    return exports
