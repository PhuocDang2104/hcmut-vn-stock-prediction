from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from vnstock.utils.io import read_yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


@lru_cache(maxsize=1)
def project_paths() -> dict[str, Path]:
    config = read_yaml(repo_root() / "configs" / "paths.yaml").get("paths", {})
    return {name: resolve_path(value) for name, value in config.items()}


def path_for(key: str) -> Path:
    paths = project_paths()
    if key not in paths:
        raise KeyError(f"Unknown path key: {key}")
    return paths[key]

