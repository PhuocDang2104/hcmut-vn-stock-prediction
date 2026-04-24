from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    return content


def write_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=str)
    return output_path


def load_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix == ".csv":
        return pd.read_csv(table_path, **kwargs)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path, **kwargs)
    raise ValueError(f"Unsupported table format: {table_path.suffix}")


def save_table(frame: pd.DataFrame, path: str | Path, index: bool = False, **kwargs: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if output_path.suffix == ".csv":
        frame.to_csv(output_path, index=index, **kwargs)
    elif output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=index, **kwargs)
    else:
        raise ValueError(f"Unsupported table format: {output_path.suffix}")
    return output_path

