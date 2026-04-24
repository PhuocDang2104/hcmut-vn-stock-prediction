from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class XLSTMTSConfig:
    context_length: int
    hidden_dim: int
    num_blocks: int
    dropout: float
    target: str


class ResidualLSTMBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent_input = self.pre_norm(x)
        recurrent_output, _ = self.lstm(recurrent_input)
        x = x + self.dropout(recurrent_output)
        return x + self.dropout(self.ffn(self.post_norm(x)))


class XLSTMTSRegressor(nn.Module):
    def __init__(
        self,
        *,
        num_features: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualLSTMBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.output_norm(hidden[:, -1, :])
        return {
            "prediction": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


def build_model(config: dict, num_features: int) -> XLSTMTSRegressor:
    return XLSTMTSRegressor(
        num_features=num_features,
        hidden_dim=int(config["hidden_dim"]),
        num_blocks=int(config["num_blocks"]),
        dropout=float(config["dropout"]),
    )


def build_model_spec(config: dict) -> dict[str, object]:
    return {
        "family": "xlstm_ts",
        "input_shape": ["batch", config["context_length"], "num_features"],
        "hidden_dim": config["hidden_dim"],
        "num_blocks": config["num_blocks"],
        "dropout": config["dropout"],
        "target": config["target"],
        "implementation": "residual_lstm_multitask_proxy",
        "direction_loss_weight": config.get("direction_loss_weight", 0.0),
    }
