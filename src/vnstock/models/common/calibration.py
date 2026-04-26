from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectionCalibration:
    score_column: str
    threshold: float
    valid_accuracy: float
    valid_balanced_accuracy: float
    default_threshold: float
    default_valid_accuracy: float
    default_valid_balanced_accuracy: float
    predicted_positive_rate: float
    optimized_metric: str = "accuracy"
    optimized_metric_value: float = float("nan")


def calibrate_direction_threshold(
    frame: pd.DataFrame,
    *,
    score_column: str,
    y_true_column: str,
    default_threshold: float,
    min_improvement: float = 0.0,
    min_positive_rate: float = 0.2,
    max_positive_rate: float = 0.8,
    optimize_metric: str = "accuracy",
) -> DirectionCalibration:
    clean = frame.dropna(subset=[score_column, y_true_column]).copy()
    if clean.empty:
        return DirectionCalibration(
            score_column=score_column,
            threshold=default_threshold,
            valid_accuracy=float("nan"),
            valid_balanced_accuracy=float("nan"),
            default_threshold=default_threshold,
            default_valid_accuracy=float("nan"),
            default_valid_balanced_accuracy=float("nan"),
            predicted_positive_rate=float("nan"),
            optimized_metric=optimize_metric,
            optimized_metric_value=float("nan"),
        )

    y_true = clean[y_true_column].to_numpy(dtype=float) > 0
    scores = clean[score_column].to_numpy(dtype=float)
    candidates = np.unique(np.r_[np.quantile(scores, np.linspace(0.01, 0.99, 99)), default_threshold])

    default_accuracy = _direction_accuracy(y_true, scores, default_threshold)
    default_balanced_accuracy = _balanced_direction_accuracy(y_true, scores, default_threshold)
    default_metric = _calibration_metric(
        y_true,
        scores,
        default_threshold,
        optimize_metric=optimize_metric,
    )
    best_threshold = default_threshold
    best_metric = default_metric
    for threshold in candidates:
        metric = _calibration_metric(
            y_true,
            scores,
            float(threshold),
            optimize_metric=optimize_metric,
        )
        if metric > best_metric or (
            metric == best_metric and abs(threshold - default_threshold) < abs(best_threshold - default_threshold)
        ):
            best_metric = metric
            best_threshold = float(threshold)

    best_positive_rate = float(np.mean(scores > best_threshold))
    if (
        best_metric < default_metric + min_improvement
        or best_positive_rate < min_positive_rate
        or best_positive_rate > max_positive_rate
    ):
        best_threshold = default_threshold
        best_metric = default_metric
        best_positive_rate = float(np.mean(scores > best_threshold))

    return DirectionCalibration(
        score_column=score_column,
        threshold=float(best_threshold),
        valid_accuracy=_direction_accuracy(y_true, scores, best_threshold),
        valid_balanced_accuracy=_balanced_direction_accuracy(y_true, scores, best_threshold),
        default_threshold=default_threshold,
        default_valid_accuracy=float(default_accuracy),
        default_valid_balanced_accuracy=float(default_balanced_accuracy),
        predicted_positive_rate=best_positive_rate,
        optimized_metric=optimize_metric,
        optimized_metric_value=float(best_metric),
    )


def _direction_accuracy(y_true_positive: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(y_true_positive == (scores > threshold)))


def _balanced_direction_accuracy(y_true_positive: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y_pred_positive = scores > threshold
    positive_mask = y_true_positive
    negative_mask = ~y_true_positive
    if not positive_mask.any() or not negative_mask.any():
        return float("nan")
    true_positive_rate = float(np.mean(y_pred_positive[positive_mask]))
    true_negative_rate = float(np.mean(~y_pred_positive[negative_mask]))
    return 0.5 * (true_positive_rate + true_negative_rate)


def _calibration_metric(
    y_true_positive: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    optimize_metric: str,
) -> float:
    if optimize_metric == "balanced_accuracy":
        return _balanced_direction_accuracy(y_true_positive, scores, threshold)
    if optimize_metric == "accuracy":
        return _direction_accuracy(y_true_positive, scores, threshold)
    raise ValueError(f"Unsupported direction calibration metric: {optimize_metric}")
