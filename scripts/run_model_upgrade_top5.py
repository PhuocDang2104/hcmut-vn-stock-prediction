from __future__ import annotations

import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
from vnstock.models.common.sequence_data import (  # noqa: E402
    SequenceSampleSet,
    build_scaled_sequence_splits,
    cap_sequence_samples,
    load_shared_feature_panel,
)
from vnstock.models.common.torch_training import fit_regression_model, predict_multitask_model  # noqa: E402
from vnstock.models.xlstm_ts.model import XLSTMTSRegressor  # noqa: E402
from vnstock.pipelines.run_investment_backtest import _markdown_table, compute_cross_section_stats  # noqa: E402
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402
from vnstock.utils.schema import PredictionContext  # noqa: E402


TOP_K = 5
SEQ_LEN = 64
TARGET = "target_ret_5d"
SEED = 42
DEVICE = torch.device("cpu")
MAX_TRAIN_SAMPLES = 20_000
MAX_VALID_SAMPLES = 6_000
BASELINE_PATH = ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "model_upgrade_top5"
PREDICTION_DIR = ROOT / "outputs" / "final" / "model_upgrade_top5"
FIGURE_DIR = ROOT / "outputs" / "figures" / "model_upgrade_top5"
FINAL_PATH = ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"

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


@dataclass(frozen=True)
class TrainSpec:
    name: str
    model: nn.Module
    direction_loss_weight: float
    rank_loss_weight: float
    epochs: int
    score_mode: str


def main() -> None:
    set_seed(SEED)
    clean_dirs()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PREDICTION_DIR)
    ensure_dir(FIGURE_DIR)

    panel = load_shared_feature_panel()
    splits, _ = build_scaled_sequence_splits(
        feature_panel=panel,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET,
        lookback=SEQ_LEN,
    )
    splits = cap_training_splits(splits)

    predictions: list[pd.DataFrame] = [load_current_hybrid()]
    specs = [
        TrainSpec(
            name="Hybrid xLSTM gated direction",
            model=XLSTMTSRegressor(
                num_features=len(FEATURE_COLUMNS),
                hidden_dim=128,
                num_blocks=3,
                dropout=0.15,
                pooling="gated_concat",
            ),
            direction_loss_weight=0.2,
            rank_loss_weight=0.0,
            epochs=2,
            score_mode="return_direction",
        ),
        TrainSpec(
            name="MultiKernel CNN-BiGRU attention",
            model=MultiKernelCNNBiGRURegressor(len(FEATURE_COLUMNS), hidden_dim=96, dropout=0.15),
            direction_loss_weight=0.2,
            rank_loss_weight=0.0,
            epochs=2,
            score_mode="return_direction",
        ),
    ]

    for spec in specs:
        predictions.append(train_candidate(spec, splits))

    metrics = pd.DataFrame([compute_metrics(frame) for frame in predictions])
    metrics["production_score"] = production_score(metrics)
    metrics = metrics.sort_values("production_score", ascending=False).reset_index(drop=True)
    save_table(metrics, OUTPUT_DIR / "upgrade_metrics.csv")

    for frame in predictions:
        name = slug(str(frame["model_family"].iloc[0]))
        save_table(frame, PREDICTION_DIR / f"{name}_predictions.parquet")

    best_name = str(metrics.iloc[0]["model"])
    best_frame = next(frame for frame in predictions if str(frame["model_family"].iloc[0]) == best_name)
    save_table(best_frame, PREDICTION_DIR / "selected_best_predictions.parquet")
    if best_name != "Hybrid xLSTM Direction-Excess Blend":
        save_table(rename_as_final(best_frame), FINAL_PATH)
    else:
        # Preserve the existing production artifact; copy a selected snapshot for traceability.
        shutil.copy2(BASELINE_PATH, PREDICTION_DIR / "selected_production_snapshot.parquet")

    write_report(metrics)
    plot_metrics(metrics)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


class MultiKernelCNNBiGRURegressor(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        branch_channels = 32
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(num_features, branch_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel in (3, 5, 9, 15)
            ]
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(branch_channels * 4, branch_channels),
            nn.GELU(),
            nn.Linear(branch_channels, branch_channels * 4),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv1d(branch_channels * 4, hidden_dim, kernel_size=1)
        self.recurrent = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        recurrent_dim = hidden_dim * 2
        self.attention = nn.Sequential(
            nn.LayerNorm(recurrent_dim),
            nn.Linear(recurrent_dim, 1),
        )
        self.norm = nn.LayerNorm(recurrent_dim * 2)
        self.return_head = _head(recurrent_dim * 2, hidden_dim, dropout)
        self.direction_head = _head(recurrent_dim * 2, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        channels_first = x.transpose(1, 2)
        multi_scale = torch.cat([branch(channels_first) for branch in self.branches], dim=1)
        gate = self.channel_gate(multi_scale).unsqueeze(-1)
        encoded = self.proj(multi_scale * gate).transpose(1, 2)
        output, _ = self.recurrent(encoded)
        last_state = output[:, -1, :]
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        attention_pool = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        pooled = self.norm(torch.cat([last_state, attention_pool], dim=-1))
        return {
            "prediction": self.return_head(pooled).squeeze(-1),
            "direction_logit": self.direction_head(pooled).squeeze(-1),
        }


def _head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def train_candidate(spec: TrainSpec, splits: dict[str, Any]) -> pd.DataFrame:
    start = time.time()
    result = fit_regression_model(
        spec.model,
        splits["train"],
        splits["valid"],
        batch_size=384,
        eval_batch_size=1024,
        epochs=spec.epochs,
        learning_rate=4e-4,
        weight_decay=1e-4,
        patience=2,
        huber_delta=0.05,
        direction_loss_weight=spec.direction_loss_weight,
        direction_large_move_quantile=0.7,
        direction_large_move_weight=1.5,
        rank_loss_weight=spec.rank_loss_weight,
        rank_loss_min_target_diff=0.005,
        checkpoint_metric="valid_loss",
        checkpoint_mode="min",
        device=DEVICE,
        input_dtype=torch.float32,
    )
    prediction = predict_multitask_model(
        spec.model,
        splits["test"],
        batch_size=1024,
        device=DEVICE,
        input_dtype=torch.float32,
    )
    frame = splits["test"].meta.copy()
    frame[TARGET] = splits["test"].y
    frame["raw_return_pred"] = prediction.y_pred
    if prediction.direction_score is not None:
        frame["direction_score"] = prediction.direction_score
    frame["y_pred"] = build_score(frame, spec.score_mode)
    context = PredictionContext(
        model_family=spec.name,
        model_version=f"{slug(spec.name)}_upgrade_v1",
        target_name=TARGET,
        horizon=5,
        run_id=f"{slug(spec.name)}_{int(start)}",
    )
    output = standardize_prediction_output(frame, context, y_true_column=TARGET)
    output["coverage_scope"] = "full_test"
    output["training_seconds"] = time.time() - start
    output["best_valid_loss"] = result.best_valid_loss
    return output


def cap_training_splits(splits: dict[str, SequenceSampleSet]) -> dict[str, SequenceSampleSet]:
    output = dict(splits)
    output["train"] = cap_sequence_samples(output["train"], MAX_TRAIN_SAMPLES, SEED)
    output["valid"] = cap_sequence_samples(output["valid"], MAX_VALID_SAMPLES, SEED)
    return output


def build_score(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode != "return_direction" or "direction_score" not in frame.columns:
        return frame["raw_return_pred"].to_numpy(dtype=np.float32)
    scored = frame[["date", "raw_return_pred", "direction_score"]].copy()
    scored["date"] = pd.to_datetime(scored["date"])
    scored["return_z"] = scored.groupby("date")["raw_return_pred"].transform(zscore)
    direction_logit = np.log(np.clip(scored["direction_score"].to_numpy(dtype=float), 1e-5, 1 - 1e-5))
    direction_logit -= np.log1p(-np.clip(scored["direction_score"].to_numpy(dtype=float), 1e-5, 1 - 1e-5))
    scored["direction_logit"] = direction_logit
    scored["direction_z"] = scored.groupby("date")["direction_logit"].transform(zscore)
    score = 0.75 * scored["return_z"].to_numpy(dtype=float) + 0.25 * scored["direction_z"].to_numpy(dtype=float)
    return score.astype(np.float32, copy=False)


def load_current_hybrid() -> pd.DataFrame:
    frame = load_table(BASELINE_PATH)
    output = frame.copy()
    output["model_family"] = "Hybrid xLSTM Direction-Excess Blend"
    output["model_version"] = "current_production"
    output["coverage_scope"] = "full_test"
    return output


def rename_as_final(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["model_family"] = "Hybrid xLSTM Direction-Excess Blend"
    output["model_version"] = "selected_upgrade_top5_v1"
    output["run_id"] = "selected_upgrade_top5_v1"
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


def production_score(metrics: pd.DataFrame) -> pd.Series:
    components = {
        "IC": 0.25,
        "RankIC": 0.25,
        "LongShort5": 0.25,
        "Top5_Return": 0.15,
        "Top5_Direction_Acc": 0.10,
    }
    score = pd.Series(0.0, index=metrics.index)
    for column, weight in components.items():
        values = metrics[column].astype(float)
        std = values.std(ddof=0)
        normalized = (values - values.mean()) / std if std and np.isfinite(std) else values * 0.0
        score = score + weight * normalized
    return score


def write_report(metrics: pd.DataFrame) -> None:
    cols = [
        "model",
        "rows",
        "IC",
        "RankIC",
        "ICIR",
        "RankICIR",
        "Direction_Acc",
        "Balanced_Acc",
        "Top5_Return",
        "LongShort5",
        "Top5_Direction_Acc",
        "production_score",
    ]
    winner = metrics.iloc[0]
    lines = [
        "# Model Upgrade Top-5 Evaluation",
        "",
        "Goal: test focused upgrades from the architecture note and keep only the strongest production candidate.",
        "",
        "Candidates:",
        "",
        "- Current `Hybrid xLSTM Direction-Excess Blend` production artifact.",
        "- `Hybrid xLSTM gated direction`: gated-concat xLSTM with direction loss. Pairwise rank loss was disabled in this CPU run because it was too slow for a clean local rerun.",
        "- `MultiKernel CNN-BiGRU attention`: multi-scale Conv1D, channel gate, BiGRU, attention pooling.",
        "",
        "Selection score:",
        "",
        "```text",
        "0.25 * z(IC) + 0.25 * z(RankIC) + 0.25 * z(LongShort5)",
        "  + 0.15 * z(Top5_Return) + 0.10 * z(Top5_Direction_Acc)",
        "```",
        "",
        _markdown_table(metrics[cols]),
        "",
        f"Selected production candidate: `{winner['model']}`.",
        "",
        "If an upgrade does not beat the current Hybrid xLSTM by production score, the existing production artifact is preserved.",
        "",
        "Artifacts:",
        "",
        "- `outputs/reports/model_upgrade_top5/upgrade_metrics.csv`",
        "- `outputs/final/model_upgrade_top5/`",
        "- `outputs/figures/model_upgrade_top5/upgrade_longshort5.png`",
    ]
    (OUTPUT_DIR / "upgrade_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(metrics["model"], metrics["LongShort5"])
    plt.title("Upgrade candidates: LongShort5")
    plt.ylabel("mean t+5 top5-bottom5 spread")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "upgrade_longshort5.png", dpi=160)
    plt.close()


def clean_dirs() -> None:
    for path in (OUTPUT_DIR, PREDICTION_DIR, FIGURE_DIR):
        if not path.exists():
            continue
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise ValueError(f"Refusing to delete outside repo: {resolved}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not std or not np.isfinite(std):
        return values * 0.0
    return (values - values.mean()) / std


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
