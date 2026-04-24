from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ArrayDatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_array_dataset(processed_dir: str | Path) -> ArrayDatasetBundle:
    root = Path(processed_dir)
    return ArrayDatasetBundle(
        X_train=np.load(root / "X_train.npy"),
        y_train=np.load(root / "y_train.npy"),
        X_valid=np.load(root / "X_valid.npy"),
        y_valid=np.load(root / "y_valid.npy"),
        X_test=np.load(root / "X_test.npy"),
        y_test=np.load(root / "y_test.npy"),
    )


def load_split(processed_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(processed_dir)
    return np.load(root / f"X_{split}.npy"), np.load(root / f"y_{split}.npy")
