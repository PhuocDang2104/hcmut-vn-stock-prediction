from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 0.05) -> float:
    error = np.abs(y_true - y_pred)
    quadratic = np.minimum(error, delta)
    linear = error - quadratic
    return float(np.mean(0.5 * np.square(quadratic) + delta * linear))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return math.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    return float(spearmanr(y_true, y_pred, nan_policy="omit").statistic)


def directional_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    return float(np.mean((y_true > 0) == (y_score > threshold)))


def balanced_directional_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    actual_up = y_true > 0
    predicted_up = y_score > threshold
    actual_down = ~actual_up
    if not actual_up.any() or not actual_down.any():
        return math.nan
    recall_up_value = float(np.mean(predicted_up[actual_up]))
    recall_down_value = float(np.mean(~predicted_up[actual_down]))
    return 0.5 * (recall_up_value + recall_down_value)


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    positive_rate = float(np.mean(y_true > 0))
    return max(positive_rate, 1.0 - positive_rate)


def precision_up(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    predicted_up = y_score > threshold
    if not predicted_up.any():
        return 0.0
    return float(np.mean(y_true[predicted_up] > 0))


def recall_up(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    actual_up = y_true > 0
    if not actual_up.any():
        return 0.0
    return float(np.mean(y_score[actual_up] > threshold))


def f1_up(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    precision_value = precision_up(y_true, y_score, threshold)
    recall_value = recall_up(y_true, y_score, threshold)
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def precision_down(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    predicted_down = y_score <= threshold
    if not predicted_down.any():
        return 0.0
    return float(np.mean(y_true[predicted_down] <= 0))


def recall_down(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    actual_down = y_true <= 0
    if not actual_down.any():
        return 0.0
    return float(np.mean(y_score[actual_down] <= threshold))


def f1_down(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.0) -> float:
    precision_value = precision_down(y_true, y_score, threshold)
    recall_value = recall_down(y_true, y_score, threshold)
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def information_coefficient(frame: pd.DataFrame) -> float:
    if frame.empty:
        return math.nan
    by_date = frame.groupby("date")[["y_true", "y_pred"]].apply(
        lambda group: spearman(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
    )
    return float(by_date.mean())


def top_k_hit_rate(frame: pd.DataFrame, k: int = 3) -> float:
    if frame.empty:
        return math.nan

    hits: list[float] = []
    for _, group in frame.groupby("date"):
        top_pred = group.nlargest(min(k, len(group)), "y_pred")
        if top_pred.empty:
            continue
        hits.append(float((top_pred["y_true"] > 0).mean()))
    return float(np.mean(hits)) if hits else math.nan


def top_k_realized_return(frame: pd.DataFrame, k: int = 3) -> float:
    if frame.empty:
        return math.nan

    realized_returns: list[float] = []
    for _, group in frame.groupby("date"):
        top_pred = group.nlargest(min(k, len(group)), "y_pred")
        if top_pred.empty:
            continue
        realized_returns.append(float(top_pred["y_true"].mean()))
    return float(np.mean(realized_returns)) if realized_returns else math.nan


def long_short_spread(frame: pd.DataFrame, k: int = 3) -> float:
    if frame.empty:
        return math.nan

    spreads: list[float] = []
    for _, group in frame.groupby("date"):
        bucket_size = min(k, len(group))
        top_pred = group.nlargest(bucket_size, "y_pred")
        bottom_pred = group.nsmallest(bucket_size, "y_pred")
        if top_pred.empty or bottom_pred.empty:
            continue
        spreads.append(float(top_pred["y_true"].mean() - bottom_pred["y_true"].mean()))
    return float(np.mean(spreads)) if spreads else math.nan


def confident_directional_accuracy(
    y_true: np.ndarray,
    direction_score: np.ndarray,
    threshold: float,
    top_fraction: float = 0.2,
) -> float:
    if len(y_true) == 0:
        return math.nan
    confidence = np.abs(direction_score - threshold)
    cutoff = np.quantile(confidence, max(0.0, min(1.0, 1.0 - top_fraction)))
    mask = confidence >= cutoff
    if not mask.any():
        return math.nan
    return directional_accuracy(y_true[mask], direction_score[mask], threshold)


def large_move_directional_accuracy(
    y_true: np.ndarray,
    direction_score: np.ndarray,
    threshold: float,
    min_abs_return: float = 0.02,
) -> float:
    mask = np.abs(y_true) >= min_abs_return
    if not mask.any():
        return math.nan
    return directional_accuracy(y_true[mask], direction_score[mask], threshold)


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    subset = frame.dropna(subset=["y_true", "y_pred"]).copy()
    if subset.empty:
        return {
            "huber": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "pearson": math.nan,
            "spearman": math.nan,
            "directional_accuracy": math.nan,
            "balanced_directional_accuracy": math.nan,
            "majority_baseline_accuracy": math.nan,
            "confident_directional_accuracy_top20": math.nan,
            "large_move_directional_accuracy_2pct": math.nan,
            "precision_up": math.nan,
            "recall_up": math.nan,
            "f1_up": math.nan,
            "precision_down": math.nan,
            "recall_down": math.nan,
            "f1_down": math.nan,
            "information_coefficient": math.nan,
            "top_k_hit_rate": math.nan,
            "top_k_realized_return": math.nan,
            "long_short_spread": math.nan,
        }

    y_true = subset["y_true"].to_numpy(dtype=float)
    y_pred = subset["y_pred"].to_numpy(dtype=float)
    direction_score_column = _direction_score_column(subset)
    direction_score = subset[direction_score_column].to_numpy(dtype=float)
    direction_threshold = _direction_threshold(subset, direction_score_column)
    return {
        "huber": huber(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, direction_score, direction_threshold),
        "balanced_directional_accuracy": balanced_directional_accuracy(y_true, direction_score, direction_threshold),
        "majority_baseline_accuracy": majority_baseline_accuracy(y_true),
        "confident_directional_accuracy_top20": confident_directional_accuracy(
            y_true,
            direction_score,
            direction_threshold,
            top_fraction=0.2,
        ),
        "large_move_directional_accuracy_2pct": large_move_directional_accuracy(
            y_true,
            direction_score,
            direction_threshold,
            min_abs_return=0.02,
        ),
        "precision_up": precision_up(y_true, direction_score, direction_threshold),
        "recall_up": recall_up(y_true, direction_score, direction_threshold),
        "f1_up": f1_up(y_true, direction_score, direction_threshold),
        "precision_down": precision_down(y_true, direction_score, direction_threshold),
        "recall_down": recall_down(y_true, direction_score, direction_threshold),
        "f1_down": f1_down(y_true, direction_score, direction_threshold),
        "direction_threshold": direction_threshold,
        "direction_positive_rate": float(np.mean(direction_score > direction_threshold)),
        "information_coefficient": information_coefficient(subset),
        "top_k_hit_rate": top_k_hit_rate(subset),
        "top_k_realized_return": top_k_realized_return(subset),
        "long_short_spread": long_short_spread(subset),
    }


def _direction_score_column(frame: pd.DataFrame) -> str:
    if "direction_score" in frame.columns:
        return "direction_score"
    return "y_pred"


def _direction_threshold(frame: pd.DataFrame, score_column: str) -> float:
    if "direction_threshold" in frame.columns and frame["direction_threshold"].notna().any():
        return float(frame["direction_threshold"].dropna().iloc[0])
    if score_column == "direction_score":
        return 0.5
    return 0.0
