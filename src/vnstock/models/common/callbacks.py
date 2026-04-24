from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best_score: float | None = None
    wait: int = 0
    stopped: bool = False

    def update(self, score: float) -> bool:
        if self.best_score is None or score < (self.best_score - self.min_delta):
            self.best_score = score
            self.wait = 0
            return False

        self.wait += 1
        self.stopped = self.wait >= self.patience
        return self.stopped

