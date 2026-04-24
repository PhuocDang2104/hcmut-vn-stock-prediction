from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectionCalibration:
    score_column: str
    threshold: float
    valid_accuracy: float
    default_threshold: float
    default_valid_accuracy: float
    predicted_positive_rate: float


def calibrate_direction_threshold(
    frame: pd.DataFrame,
    *,
    score_column: str,
    y_true_column: str,
    default_threshold: float,
    min_improvement: float = 0.0,
    min_positive_rate: float = 0.2,
    max_positive_rate: float = 0.8,
) -> DirectionCalibration:
    clean = frame.dropna(subset=[score_column, y_true_column]).copy()
    if clean.empty:
        return DirectionCalibration(
            score_column=score_column,
            threshold=default_threshold,
            valid_accuracy=float("nan"),
            default_threshold=default_threshold,
            default_valid_accuracy=float("nan"),
            predicted_positive_rate=float("nan"),
        )

    y_true = clean[y_true_column].to_numpy(dtype=float) > 0
    scores = clean[score_column].to_numpy(dtype=float)
    candidates = np.unique(np.r_[np.quantile(scores, np.linspace(0.01, 0.99, 99)), default_threshold])

    default_accuracy = _direction_accuracy(y_true, scores, default_threshold)
    best_threshold = default_threshold
    best_accuracy = default_accuracy
    for threshold in candidates:
        accuracy = _direction_accuracy(y_true, scores, float(threshold))
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and abs(threshold - default_threshold) < abs(best_threshold - default_threshold)
        ):
            best_accuracy = accuracy
            best_threshold = float(threshold)

    best_positive_rate = float(np.mean(scores > best_threshold))
    if (
        best_accuracy < default_accuracy + min_improvement
        or best_positive_rate < min_positive_rate
        or best_positive_rate > max_positive_rate
    ):
        best_threshold = default_threshold
        best_accuracy = default_accuracy
        best_positive_rate = float(np.mean(scores > best_threshold))

    return DirectionCalibration(
        score_column=score_column,
        threshold=float(best_threshold),
        valid_accuracy=float(best_accuracy),
        default_threshold=default_threshold,
        default_valid_accuracy=float(default_accuracy),
        predicted_positive_rate=best_positive_rate,
    )


def _direction_accuracy(y_true_positive: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(y_true_positive == (scores > threshold)))
