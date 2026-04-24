from __future__ import annotations

import pandas as pd
import seaborn as sns


def plot_residual_histogram(frame: pd.DataFrame):
    residuals = frame["y_pred"] - frame["y_true"]
    ax = sns.histplot(residuals, kde=True)
    ax.set_title("Residual Distribution")
    return ax


def plot_leaderboard(frame: pd.DataFrame, metric: str = "rmse"):
    leaderboard = frame.sort_values(metric, ascending=True)
    ax = sns.barplot(data=leaderboard, x=metric, y="model_family")
    ax.set_title(f"Leaderboard by {metric.upper()}")
    return ax

