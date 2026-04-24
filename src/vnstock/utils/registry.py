from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from vnstock.utils.io import ensure_dir, write_json
from vnstock.utils.paths import path_for


def generate_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def experiment_manifest_path(run_id: str) -> Path:
    return path_for("registry_root") / "experiments" / f"{run_id}.json"


def model_registry_dir(model_name: str) -> Path:
    directory = path_for("registry_root") / "models" / model_name
    ensure_dir(directory)
    return directory


def save_run_manifest(run_id: str, payload: dict[str, Any]) -> Path:
    return write_json(payload, experiment_manifest_path(run_id))

