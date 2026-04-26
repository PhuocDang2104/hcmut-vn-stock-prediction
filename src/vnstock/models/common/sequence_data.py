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


def build_effective_feature_columns(
    feature_columns: list[str],
    wavelet_config: dict[str, Any] | None = None,
) -> list[str]:
    if not _wavelet_enabled(wavelet_config):
        return list(feature_columns)

    denoise_columns = _resolve_wavelet_columns(feature_columns, wavelet_config or {})
    suffix = str((wavelet_config or {}).get("suffix", "_denoised"))
    denoised_names = [f"{column}{suffix}" for column in denoise_columns]
    if bool((wavelet_config or {}).get("append", True)):
        return [*feature_columns, *denoised_names]
    return denoised_names


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
    wavelet_config: dict[str, Any] | None = None,
    target_scaling: dict[str, Any] | None = None,
) -> tuple[dict[str, SequenceSampleSet], dict[str, dict[str, list[float]]]]:
    scalers = fit_symbol_zscore_scalers(feature_panel, feature_columns)
    target_scalers = fit_symbol_target_scalers(feature_panel, target_column) if _target_scaling_enabled(target_scaling) else {}
    effective_feature_columns = build_effective_feature_columns(feature_columns, wavelet_config)
    buckets: dict[str, dict[str, Any]] = {
        split_name: {"X": [], "y": [], "meta": []} for split_name in ("train", "valid", "test")
    }

    for symbol, group in feature_panel.groupby("symbol", sort=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        symbol_key = str(symbol)
        if symbol_key not in scalers:
            continue
        if _target_scaling_enabled(target_scaling) and symbol_key not in target_scalers:
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
            window = _apply_wavelet_to_window(
                window,
                feature_columns=feature_columns,
                wavelet_config=wavelet_config,
            )
            target_raw = targets[end_idx]
            target_value = target_raw
            meta_extra: dict[str, float] = {}
            if _target_scaling_enabled(target_scaling):
                target_mean = float(target_scalers[symbol_key]["mean"])
                target_std = float(target_scalers[symbol_key]["std"])
                target_value = (target_raw - target_mean) / target_std
                meta_extra = {
                    "target_raw": float(target_raw),
                    "target_mean": target_mean,
                    "target_std": target_std,
                }
            if np.isnan(window).any() or np.isnan(target_value):
                continue
            buckets[split_name]["X"].append(window.astype(np.float32, copy=False))
            buckets[split_name]["y"].append(float(target_value))
            buckets[split_name]["meta"].append(
                {
                    "symbol": symbol_key,
                    "date": ordered.loc[end_idx, "date"],
                    "split": split_name,
                    **meta_extra,
                }
            )

    if target_scalers:
        scalers = {**scalers, "__target__": target_scalers}
    return _finalize_sequence_buckets(buckets, lookback, len(effective_feature_columns), dtype=np.float32), scalers


def fit_symbol_target_scalers(
    feature_panel: pd.DataFrame,
    target_column: str,
) -> dict[str, dict[str, float]]:
    scalers: dict[str, dict[str, float]] = {}
    train_frame = feature_panel.loc[feature_panel["split"] == "train", ["symbol", target_column]].copy()
    for symbol, group in train_frame.groupby("symbol"):
        values = group[target_column].dropna().to_numpy(dtype=np.float32)
        if len(values) == 0:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not np.isfinite(std) or std <= 1e-6:
            std = 1.0
        scalers[str(symbol)] = {"mean": mean, "std": std}
    return scalers


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


def _wavelet_enabled(wavelet_config: dict[str, Any] | None) -> bool:
    return bool(wavelet_config and wavelet_config.get("enabled", False))


def _target_scaling_enabled(target_scaling: dict[str, Any] | None) -> bool:
    return bool(target_scaling and target_scaling.get("enabled", False))


def _resolve_wavelet_columns(
    feature_columns: list[str],
    wavelet_config: dict[str, Any],
) -> list[str]:
    configured = wavelet_config.get("feature_columns") or feature_columns
    selected = [str(column) for column in configured if str(column) in feature_columns]
    if not selected:
        raise ValueError("wavelet_denoise.feature_columns did not match any input feature columns.")
    return selected


def _apply_wavelet_to_window(
    window: np.ndarray,
    *,
    feature_columns: list[str],
    wavelet_config: dict[str, Any] | None,
) -> np.ndarray:
    if not _wavelet_enabled(wavelet_config):
        return window

    config = wavelet_config or {}
    denoise_columns = _resolve_wavelet_columns(feature_columns, config)
    denoise_indices = [feature_columns.index(column) for column in denoise_columns]
    denoised = np.empty((window.shape[0], len(denoise_indices)), dtype=np.float32)
    for output_idx, feature_idx in enumerate(denoise_indices):
        denoised[:, output_idx] = wavelet_denoise_window(
            window[:, feature_idx],
            wavelet=str(config.get("wavelet", "db4")),
            level=int(config.get("level", 2)),
            mode=str(config.get("mode", "symmetric")),
            threshold_mode=str(config.get("threshold_mode", "soft")),
        )

    if bool(config.get("append", True)):
        return np.concatenate([window, denoised], axis=1).astype(np.float32, copy=False)
    return denoised.astype(np.float32, copy=False)


def wavelet_denoise_window(
    values: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 2,
    mode: str = "symmetric",
    threshold_mode: str = "soft",
) -> np.ndarray:
    try:
        import pywt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyWavelets is required for causal wavelet denoising. "
            "Install it with `pip install PyWavelets`."
        ) from exc

    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("wavelet_denoise_window expects a 1D array.")
    if len(x) < 4 or not np.isfinite(x).all():
        return x.astype(np.float32, copy=False)

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(data_len=len(x), filter_len=wavelet_obj.dec_len)
    safe_level = max(1, min(int(level), max_level))
    coeffs = pywt.wavedec(x, wavelet=wavelet_obj, mode=mode, level=safe_level)
    if len(coeffs) <= 1:
        return x.astype(np.float32, copy=False)

    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    if not np.isfinite(sigma) or sigma <= 0:
        return x.astype(np.float32, copy=False)
    threshold = sigma * np.sqrt(2.0 * np.log(len(x)))
    denoised_coeffs = [coeffs[0]]
    for coeff in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(coeff, threshold, mode=threshold_mode))

    reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet_obj, mode=mode)
    return reconstructed[: len(x)].astype(np.float32, copy=False)
