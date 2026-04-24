from __future__ import annotations

from vnstock.models.common.base_trainer import BaseTrainer, TrainingArtifacts


class PatchTSTTrainer(BaseTrainer):
    def __init__(self, config: dict) -> None:
        super().__init__(model_name="patchtst", config=config)

    def fit(self) -> TrainingArtifacts:
        return self.record_scaffold_state(
            note="PatchTST scaffold is ready. Attach sequence patching and model wrapper here.",
            details={"expected_input_dir": "data/processed/patchtst"},
        )
