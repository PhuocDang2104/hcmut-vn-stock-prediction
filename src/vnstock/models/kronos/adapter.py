from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnstock.data.dataset_builder import export_kronos_dataset


def export_adapter_inputs(feature_panel: pd.DataFrame) -> dict[str, Path]:
    return export_kronos_dataset(feature_panel)

