from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vnstock.utils.io import load_table, read_json
from vnstock.utils.paths import path_for


@dataclass
class SequenceSampleSet:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame

    @property
    def size(self) -> int:
        return int(len(self.y))


def load_shared_feature_panel(path: str | Path | None = None) -> pd.DataFrame:
    panel_path = Path(path) if path is not None else path_for("shared_root") / "feature_panel.parquet"
    return load_table(panel_path)


def load_shared_metadata(path: str | Path | None = None) -> dict[str, Any]:
    metadata_path = Path(path) if path is not None else path_for("shared_root") / "split_meta.json"
    return read_json(metadata_path)


def resolve_feature_columns(
    config: dict[str, Any],
    metadata: dict[str, Any],
    key: str = "feature_columns",
) -> list[str]:
    configured = config.get(key)
    if configured:
        return list(configured)
    return list(metadata.get(key, []))


def fit_symbol_zscore_scalers(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, list[float]]]:
    scalers: dict[str, dict[str, list[float]]] = {}
    train_frame = feature_panel.loc[feature_panel["split"] == "train", ["symbol", *feature_columns]].copy()

    for symbol, group in train_frame.groupby("symbol"):
        means = group[feature_columns].mean().to_numpy(dtype=np.float32)
        stds = group[feature_columns].std(ddof=0).to_numpy(dtype=np.float32)
        stds = np.where(np.isfinite(stds) & (stds > 1e-6), stds, 1.0)
        scalers[str(symbol)] = {
            "mean": means.tolist(),
            "std": stds.tolist(),
        }
    return scalers


def build_scaled_sequence_splits(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
) -> tuple[dict[str, SequenceSampleSet], dict[str, dict[str, list[float]]]]:
    scalers = fit_symbol_zscore_scalers(feature_panel, feature_columns)
    buckets: dict[str, dict[str, Any]] = {
        split_name: {"X": [], "y": [], "meta": []} for split_name in ("train", "valid", "test")
    }

    for symbol, group in feature_panel.groupby("symbol", sort=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        symbol_key = str(symbol)
        if symbol_key not in scalers:
            continue

        mean = np.asarray(scalers[symbol_key]["mean"], dtype=np.float32)
        std = np.asarray(scalers[symbol_key]["std"], dtype=np.float32)
        values = ordered[feature_columns].to_numpy(dtype=np.float32)
        values = (values - mean) / std
        targets = ordered[target_column].to_numpy(dtype=np.float32)
        splits = ordered["split"].astype(str).to_numpy()

        for end_idx in range(lookback - 1, len(ordered)):
            split_name = splits[end_idx]
            if split_name not in buckets:
                continue
            window = values[end_idx - lookback + 1 : end_idx + 1]
            target_value = targets[end_idx]
            if np.isnan(window).any() or np.isnan(target_value):
                continue
            buckets[split_name]["X"].append(window.astype(np.float32, copy=False))
            buckets[split_name]["y"].append(float(target_value))
            buckets[split_name]["meta"].append(
                {
                    "symbol": symbol_key,
                    "date": ordered.loc[end_idx, "date"],
                    "split": split_name,
                }
            )

    return _finalize_sequence_buckets(buckets, lookback, len(feature_columns), dtype=np.float32), scalers


def fit_quantile_token_bins(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
    num_bins: int,
) -> dict[str, list[float]]:
    quantile_levels = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    train_frame = feature_panel.loc[feature_panel["split"] == "train", feature_columns]
    token_bins: dict[str, list[float]] = {}

    for column in feature_columns:
        values = train_frame[column].to_numpy(dtype=np.float32)
        cut_points = np.quantile(values, quantile_levels)
        unique_points = np.unique(cut_points.astype(np.float32))
        token_bins[column] = unique_points.tolist()
    return token_bins


def build_tokenized_sequence_splits(
    feature_panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
    num_bins: int,
) -> tuple[dict[str, SequenceSampleSet], dict[str, list[float]]]:
    token_bins = fit_quantile_token_bins(feature_panel, feature_columns, num_bins=num_bins)
    buckets: dict[str, dict[str, Any]] = {
        split_name: {"X": [], "y": [], "meta": []} for split_name in ("train", "valid", "test")
    }

    for symbol, group in feature_panel.groupby("symbol", sort=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        values = ordered[feature_columns].to_numpy(dtype=np.float32)
        token_values = np.zeros_like(values, dtype=np.int64)
        for feature_idx, column in enumerate(feature_columns):
            edges = np.asarray(token_bins[column], dtype=np.float32)
            token_values[:, feature_idx] = np.digitize(values[:, feature_idx], edges, right=False)

        targets = ordered[target_column].to_numpy(dtype=np.float32)
        splits = ordered["split"].astype(str).to_numpy()

        for end_idx in range(lookback - 1, len(ordered)):
            split_name = splits[end_idx]
            if split_name not in buckets:
                continue
            window = token_values[end_idx - lookback + 1 : end_idx + 1]
            target_value = targets[end_idx]
            if np.isnan(target_value):
                continue
            buckets[split_name]["X"].append(window.astype(np.int64, copy=False))
            buckets[split_name]["y"].append(float(target_value))
            buckets[split_name]["meta"].append(
                {
                    "symbol": str(symbol),
                    "date": ordered.loc[end_idx, "date"],
                    "split": split_name,
                }
            )

    return _finalize_sequence_buckets(buckets, lookback, len(feature_columns), dtype=np.int64), token_bins


def cap_sequence_samples(
    sample_set: SequenceSampleSet,
    max_samples: int | None,
    seed: int,
) -> SequenceSampleSet:
    if max_samples is None or sample_set.size <= max_samples:
        return sample_set

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(sample_set.size, size=max_samples, replace=False))
    capped_meta = sample_set.meta.iloc[indices].reset_index(drop=True)
    return SequenceSampleSet(
        X=sample_set.X[indices],
        y=sample_set.y[indices],
        meta=capped_meta,
    )


def _finalize_sequence_buckets(
    buckets: dict[str, dict[str, Any]],
    lookback: int,
    num_features: int,
    dtype: np.dtype[Any],
) -> dict[str, SequenceSampleSet]:
    sequence_sets: dict[str, SequenceSampleSet] = {}
    for split_name, bucket in buckets.items():
        x_values = bucket["X"]
        y_values = bucket["y"]
        if x_values:
            X = np.stack(x_values).astype(dtype, copy=False)
            y = np.asarray(y_values, dtype=np.float32)
        else:
            X = np.empty((0, lookback, num_features), dtype=dtype)
            y = np.empty((0,), dtype=np.float32)
        meta = pd.DataFrame(bucket["meta"])
        if not meta.empty:
            meta["date"] = pd.to_datetime(meta["date"])
        sequence_sets[split_name] = SequenceSampleSet(X=X, y=y, meta=meta)
    return sequence_sets
