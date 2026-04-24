from __future__ import annotations

import math

import numpy as np
import pandas as pd


def proxy_returns(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values("date").copy()
    return np.sign(ordered["y_pred"].to_numpy(dtype=float)) * ordered["y_true"].to_numpy(dtype=float)


def cumulative_return(frame: pd.DataFrame) -> float:
    returns = proxy_returns(frame)
    return float(np.nansum(returns))


def win_rate(frame: pd.DataFrame) -> float:
    returns = proxy_returns(frame)
    if len(returns) == 0:
        return math.nan
    return float(np.mean(returns > 0))


def max_drawdown(frame: pd.DataFrame) -> float:
    returns = proxy_returns(frame)
    if len(returns) == 0:
        return math.nan
    equity_curve = returns.cumsum()
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    return float(drawdown.min())


def sharpe_proxy(frame: pd.DataFrame) -> float:
    returns = proxy_returns(frame)
    if len(returns) < 2:
        return math.nan
    std = float(np.std(returns))
    if std == 0:
        return math.nan
    return float(np.mean(returns) / std)


def compute_backtest_proxy(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "cumulative_return": cumulative_return(frame),
        "win_rate": win_rate(frame),
        "max_drawdown": max_drawdown(frame),
        "sharpe_proxy": sharpe_proxy(frame),
    }

