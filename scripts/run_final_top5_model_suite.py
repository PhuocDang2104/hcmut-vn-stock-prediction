from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vnstock.evaluation.metrics import (  # noqa: E402
    balanced_directional_accuracy,
    directional_accuracy,
    f1_down,
    f1_up,
    majority_baseline_accuracy,
)
from vnstock.models.common.base_predictor import standardize_prediction_output  # noqa: E402
from vnstock.models.common.sequence_data import build_scaled_sequence_splits, load_shared_feature_panel  # noqa: E402
from vnstock.models.common.torch_training import (  # noqa: E402
    fit_regression_model,
    predict_multitask_model,
)
from vnstock.pipelines.run_investment_backtest import (  # noqa: E402
    _markdown_table,
    compute_cross_section_stats,
)
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402
from vnstock.utils.schema import PredictionContext  # noqa: E402


TOP_K = 5
SEQ_LEN = 64
TARGET = "target_ret_5d"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "final_top5_model_suite"
PREDICTION_DIR = ROOT / "outputs" / "final" / "model_suite_top5"
FIGURE_DIR = ROOT / "outputs" / "figures" / "final_top5_model_suite"
BEST_INPUT = ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "log_volume",
    "volume_zscore_20",
    "hl_spread",
    "oc_change",
    "gap_rel",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "rolling_vol_5",
    "rolling_vol_20",
    "vol_ratio_5_20",
    "ma_ratio_5_20",
    "open_rel",
    "high_rel",
    "low_rel",
    "ma5_rel",
    "ma20_rel",
    "rsi_14",
    "macd_hist",
    "market_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_vol_20",
    "excess_ret_1d",
    "excess_ret_5d",
    "excess_ret_20d",
    "relative_strength_5d",
    "relative_strength_20d",
]


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PREDICTION_DIR)
    ensure_dir(FIGURE_DIR)
    set_seed(42)

    panel = load_shared_feature_panel()
    splits, _ = build_scaled_sequence_splits(
        feature_panel=panel,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET,
        lookback=SEQ_LEN,
    )

    predictions = [load_final_xlstm_blend()]
    predictions.append(train_torch_model("CNN-LSTM", CNNLSTMRegressor(len(FEATURE_COLUMNS)), splits, epochs=4))
    predictions.append(train_torch_model("TCN", TCNRegressor(len(FEATURE_COLUMNS)), splits, epochs=4))
    predictions.append(train_torch_model("PatchTST", PatchTSTRegressor(len(FEATURE_COLUMNS)), splits, epochs=4))
    predictions.append(train_lightgbm_style(panel))
    kronos_frame = load_kronos_reference()
    if kronos_frame is not None:
        predictions.append(kronos_frame)

    for frame in predictions:
        name = slug(str(frame["model_family"].iloc[0]))
        save_table(frame, PREDICTION_DIR / f"{name}_predictions.parquet")

    metrics = pd.DataFrame([compute_metrics(frame) for frame in predictions])
    metrics = metrics.sort_values(["coverage_scope", "LongShort5"], ascending=[True, False])
    save_table(metrics, OUTPUT_DIR / "top5_model_suite_metrics.csv")
    write_report(metrics)
    plot_comparison(metrics)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


class CNNLSTMRegressor(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.return_head = _head(hidden_dim, dropout)
        self.direction_head = _head(hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(hidden)
        pooled = self.norm(output[:, -1, :])
        return {
            "prediction": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[..., : x.shape[-1]]
        return self.norm(x + out)


class TCNRegressor(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Conv1d(num_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, dilation=4, dropout=dropout),
            TCNBlock(hidden_dim, dilation=8, dropout=dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.return_head = _head(hidden_dim, dropout)
        self.direction_head = _head(hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.proj(x.transpose(1, 2))).transpose(1, 2)
        pooled = self.norm(hidden[:, -1, :])
        return {
            "prediction": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int = SEQ_LEN,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 96,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        patch_count = (seq_len - patch_len) // stride + 1
        self.patch_projection = nn.Linear(num_features * patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, patch_count, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.return_head = _head(d_model, dropout)
        self.direction_head = _head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(start_dim=2)
        tokens = self.patch_projection(patches) + self.position[:, : patches.shape[1], :]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return {
            "prediction": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


def _head(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def train_torch_model(name: str, model: nn.Module, splits: dict[str, Any], *, epochs: int) -> pd.DataFrame:
    start = time.time()
    result = fit_regression_model(
        model,
        splits["train"],
        splits["valid"],
        batch_size=512,
        eval_batch_size=1024,
        epochs=epochs,
        learning_rate=5e-4,
        weight_decay=1e-4,
        patience=2,
        huber_delta=0.05,
        direction_loss_weight=0.1,
        checkpoint_metric="valid_loss",
        checkpoint_mode="min",
        device=torch.device("cpu"),
        input_dtype=torch.float32,
    )
    pred = predict_multitask_model(
        model,
        splits["test"],
        batch_size=1024,
        device=torch.device("cpu"),
        input_dtype=torch.float32,
    )
    frame = splits["test"].meta.copy()
    frame[TARGET] = splits["test"].y
    frame["y_pred"] = pred.y_pred
    context = PredictionContext(
        model_family=name,
        model_version=f"{slug(name)}_top5_v1",
        target_name=TARGET,
        horizon=5,
        run_id=f"{slug(name)}_{int(start)}",
    )
    output = standardize_prediction_output(frame, context, y_true_column=TARGET)
    output["coverage_scope"] = "full_test"
    output["training_seconds"] = time.time() - start
    output["best_valid_loss"] = result.best_valid_loss
    return output


def train_lightgbm_style(panel: pd.DataFrame) -> pd.DataFrame:
    start = time.time()
    features = FEATURE_COLUMNS
    train = panel.loc[panel["split"].astype(str).eq("train")].dropna(subset=[TARGET, *features]).copy()
    test = panel.loc[panel["split"].astype(str).eq("test")].dropna(subset=[TARGET, *features]).copy()

    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        family = "LightGBM"
    except ModuleNotFoundError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.001,
            random_state=42,
        )
        family = "LightGBM-style HGBR"

    model.fit(train[features].to_numpy(dtype=np.float32), train[TARGET].to_numpy(dtype=np.float32))
    frame = test[["symbol", "date", "split", TARGET]].copy()
    frame["y_pred"] = model.predict(test[features].to_numpy(dtype=np.float32))
    context = PredictionContext(
        model_family=family,
        model_version="tabular_top5_v1",
        target_name=TARGET,
        horizon=5,
        run_id=f"lightgbm_style_{int(start)}",
    )
    output = standardize_prediction_output(frame, context, y_true_column=TARGET)
    output["coverage_scope"] = "full_test"
    output["training_seconds"] = time.time() - start
    return output


def load_final_xlstm_blend() -> pd.DataFrame:
    frame = load_table(BEST_INPUT)
    output = frame.copy()
    output["model_family"] = "Hybrid xLSTM Direction-Excess Blend"
    output["model_version"] = "return_direction_excess_top5_v1"
    output["coverage_scope"] = "full_test"
    return output


def load_kronos_reference() -> pd.DataFrame | None:
    full_path = ROOT / "outputs" / "final" / "kronos_full_test_predictions.parquet"
    latest_path = ROOT / "outputs" / "final" / "model_compare_top5" / "kronos_zero_shot_predictions.parquet"
    path = full_path if full_path.exists() else latest_path
    if not path.exists():
        return None
    frame = load_table(path)
    output = frame.copy()
    output["model_family"] = "Kronos zero-shot"
    output["coverage_scope"] = "full_test" if path == full_path else "latest_only"
    return output


def compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    test = frame.loc[frame["split"].astype(str).eq("test")].dropna(subset=["date", "symbol", "y_true", "y_pred"]).copy()
    test["date"] = pd.to_datetime(test["date"])
    cross_section = compute_cross_section_stats(test, score_column="y_pred", top_k=TOP_K)
    y_true = test["y_true"].to_numpy(dtype=float)
    y_score = test["y_pred"].to_numpy(dtype=float)
    threshold = float(test["direction_threshold"].dropna().iloc[0]) if "direction_threshold" in test.columns and test["direction_threshold"].notna().any() else 0.0
    return {
        "model": str(test["model_family"].iloc[0]) if not test.empty else "unknown",
        "coverage_scope": str(test["coverage_scope"].iloc[0]) if "coverage_scope" in test.columns and not test.empty else "full_test",
        "rows": int(len(test)),
        "dates": int(test["date"].nunique()) if not test.empty else 0,
        "symbols": int(test["symbol"].nunique()) if not test.empty else 0,
        "IC": cross_section["IC"],
        "RankIC": cross_section["RankIC"],
        "ICIR": cross_section["ICIR"],
        "RankICIR": cross_section["RankICIR"],
        "Direction_Acc": directional_accuracy(y_true, y_score, threshold) if len(test) else math.nan,
        "Balanced_Acc": balanced_directional_accuracy(y_true, y_score, threshold) if len(test) else math.nan,
        "Majority": majority_baseline_accuracy(y_true) if len(test) else math.nan,
        "F1_up": f1_up(y_true, y_score, threshold) if len(test) else math.nan,
        "F1_down": f1_down(y_true, y_score, threshold) if len(test) else math.nan,
        "Top5_Return": cross_section["TopK_Return"],
        "Bottom5_Return": cross_section["BottomK_Return"],
        "LongShort5": cross_section["LongShort"],
        "Top5_Direction_Acc": cross_section["TopK_Direction_Acc"],
    }


def write_report(metrics: pd.DataFrame) -> None:
    cols = [
        "model",
        "coverage_scope",
        "rows",
        "IC",
        "RankIC",
        "Direction_Acc",
        "Balanced_Acc",
        "Top5_Return",
        "LongShort5",
        "Top5_Direction_Acc",
    ]
    lines = [
        "# Final Top-5 Model Suite",
        "",
        "iTransformer is intentionally removed from this final suite.",
        "Kronos is loaded as the current zero-shot reference and is not retrained.",
        "",
        _markdown_table(metrics[cols]),
        "",
        "## Artifacts",
        "",
        "- `outputs/reports/final_top5_model_suite/top5_model_suite_metrics.csv`",
        "- `outputs/final/model_suite_top5/`",
        "- `outputs/figures/final_top5_model_suite/top5_model_suite_longshort.png`",
    ]
    (OUTPUT_DIR / "top5_model_suite_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(metrics: pd.DataFrame) -> None:
    full = metrics.copy()
    plt.figure(figsize=(12, 5))
    plt.bar(full["model"], full["LongShort5"])
    plt.title("Final top-5 suite: LongShort5")
    plt.ylabel("mean t+5 long-short spread")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top5_model_suite_longshort.png", dpi=160)
    plt.close()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
