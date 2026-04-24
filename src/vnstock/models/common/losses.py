from __future__ import annotations

import numpy as np


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse_loss(y_true, y_pred)))

