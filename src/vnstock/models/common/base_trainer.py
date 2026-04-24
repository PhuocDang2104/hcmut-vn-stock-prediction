from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vnstock.utils.io import ensure_dir, write_json
from vnstock.utils.registry import generate_run_id, model_registry_dir, save_run_manifest


@dataclass
class TrainingArtifacts:
    model_name: str
    run_id: str
    checkpoint_dir: Path
    manifest_path: Path
    status: str
    details: dict[str, Any] = field(default_factory=dict)


class BaseTrainer:
    def __init__(self, model_name: str, config: dict[str, Any]) -> None:
        self.model_name = model_name
        self.config = config

    def fit(self, *args: Any, **kwargs: Any) -> TrainingArtifacts:
        raise NotImplementedError("Implement framework-specific training logic in the model trainer.")

    def record_scaffold_state(self, note: str, details: dict[str, Any] | None = None) -> TrainingArtifacts:
        run_id, checkpoint_dir = self.initialize_run()
        return self.finalize_run(
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            status="scaffold_ready",
            note=note,
            details=details,
        )

    def initialize_run(self) -> tuple[str, Path]:
        run_id = generate_run_id(self.model_name)
        checkpoint_dir = ensure_dir(model_registry_dir(self.model_name) / run_id)
        return run_id, checkpoint_dir

    def finalize_run(
        self,
        run_id: str,
        checkpoint_dir: Path,
        status: str,
        note: str,
        details: dict[str, Any] | None = None,
    ) -> TrainingArtifacts:
        manifest = {
            "model_name": self.model_name,
            "run_id": run_id,
            "status": status,
            "note": note,
            "config": self.config,
            "details": details or {},
        }
        manifest_path = write_json(manifest, checkpoint_dir / "manifest.json")
        save_run_manifest(run_id, manifest)
        return TrainingArtifacts(
            model_name=self.model_name,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            manifest_path=manifest_path,
            status=status,
            details=manifest["details"],
        )
