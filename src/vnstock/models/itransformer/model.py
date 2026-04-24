from __future__ import annotations

import torch
from torch import nn


class ITransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_projection = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variate_tokens = self.sequence_projection(x.transpose(1, 2))
        encoded = self.encoder(variate_tokens)
        pooled = self.output_norm(encoded.mean(dim=1))
        return {
            "prediction": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


def build_model(config: dict) -> ITransformerRegressor:
    return ITransformerRegressor(
        seq_len=int(config["seq_len"]),
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )


def build_model_spec(config: dict) -> dict[str, object]:
    return {
        "family": "itransformer",
        "seq_len": config["seq_len"],
        "pred_len": config["pred_len"],
        "d_model": config["d_model"],
        "n_heads": config["n_heads"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        "target": config["target"],
        "implementation": "inverted_transformer_multitask_regressor",
        "direction_loss_weight": config.get("direction_loss_weight", 0.0),
    }
