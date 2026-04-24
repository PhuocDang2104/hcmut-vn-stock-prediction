from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_actual_vs_pred(frame: pd.DataFrame, symbol: str | None = None):
    subset = frame.copy()
    if symbol is not None:
        subset = subset.loc[subset["symbol"] == symbol]
    subset = subset.sort_values("date")
    melted = subset.melt(id_vars=["date"], value_vars=["y_true", "y_pred"], var_name="series", value_name="value")
    ax = sns.lineplot(data=melted, x="date", y="value", hue="series")
    ax.set_title("Actual vs Predicted")
    plt.xticks(rotation=45)
    return ax

