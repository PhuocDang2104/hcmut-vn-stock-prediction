from __future__ import annotations

from vnstock.models.common.base_trainer import BaseTrainer, TrainingArtifacts


class KronosTrainer(BaseTrainer):
    def __init__(self, config: dict) -> None:
        super().__init__(model_name="kronos", config=config)

    def fit(self) -> TrainingArtifacts:
        return self.record_scaffold_state(
            note=(
                "Kronos is intentionally not finetuned in this repo. Use the official pretrained "
                "foundation model zero-shot through KronosPredictor, then normalize its forecast "
                "back to the shared prediction contract."
            ),
            details={
                "mode": "zero_shot_foundation_required",
                "official_repo": "https://github.com/shiyu-coder/Kronos",
                "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                "model_options": ["NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"],
                "required_api": "KronosPredictor.predict",
                "local_module_available": False,
            },
        )
