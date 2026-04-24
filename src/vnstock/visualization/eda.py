from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_close_history(frame: pd.DataFrame, symbol: str):
    subset = frame.loc[frame["symbol"] == symbol].sort_values("date")
    ax = sns.lineplot(data=subset, x="date", y="close")
    ax.set_title(f"Close Price History - {symbol}")
    plt.xticks(rotation=45)
    return ax


def plot_missing_summary(frame: pd.DataFrame):
    summary = frame.isna().mean().sort_values(ascending=False).reset_index()
    summary.columns = ["column", "missing_ratio"]
    ax = sns.barplot(data=summary, x="missing_ratio", y="column")
    ax.set_title("Missing Ratio by Column")
    return ax

