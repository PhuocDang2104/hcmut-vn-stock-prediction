from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_split(processed_dir: str | Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(processed_dir) / f"{split}.parquet")

