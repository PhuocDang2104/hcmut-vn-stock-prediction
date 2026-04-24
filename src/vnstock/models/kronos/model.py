from __future__ import annotations

import torch
from torch import nn


class KronosRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_length: int,
        num_features: int,
        num_bins: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.num_features = num_features
        self.num_bins = num_bins
        self.token_embedding = nn.Embedding(num_bins * num_features, d_model)
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.time_embedding = nn.Embedding(context_length, d_model)
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
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x.shape
        if seq_len != self.context_length:
            raise ValueError(f"Expected context_length={self.context_length}, got {seq_len}")
        if num_features != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got {num_features}")

        feature_ids = torch.arange(self.num_features, device=x.device).view(1, 1, self.num_features)
        time_ids = torch.arange(self.context_length, device=x.device).view(1, self.context_length)
        token_ids = x + feature_ids * self.num_bins

        embedded = self.token_embedding(token_ids) + self.feature_embedding(feature_ids)
        step_tokens = embedded.mean(dim=2) + self.time_embedding(time_ids)
        encoded = self.encoder(step_tokens, mask=self._causal_mask(seq_len, x.device))
        pooled = self.output_norm(encoded[:, -1, :])
        return self.head(pooled).squeeze(-1)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((length, length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)


def build_model(config: dict, num_features: int) -> KronosRegressor:
    return KronosRegressor(
        context_length=int(config["context_length"]),
        num_features=num_features,
        num_bins=int(config["num_bins"]),
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
