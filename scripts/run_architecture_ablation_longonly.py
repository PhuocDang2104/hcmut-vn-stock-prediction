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
from torch.utils.data import DataLoader, TensorDataset


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
from vnstock.models.common.sequence_data import SequenceSampleSet, build_scaled_sequence_splits, load_shared_feature_panel  # noqa: E402
from vnstock.models.xlstm_ts.model import ResidualLSTMBlock  # noqa: E402
from vnstock.pipelines.run_investment_backtest import _markdown_table, compute_cross_section_stats, run_backtest  # noqa: E402
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402
from vnstock.utils.schema import PredictionContext  # noqa: E402


SEQ_LEN = 64
TOP_K = 5
SEED = 42
DEVICE = torch.device("cpu")
TRAIN_SAMPLE_CAP = 60_000
VALID_SAMPLE_CAP = 12_000
EPOCHS = 4
OUTPUT_DIR = ROOT / "outputs" / "reports" / "architecture_ablation_longonly"
PREDICTION_DIR = ROOT / "outputs" / "final" / "architecture_ablation_longonly"
FIGURE_DIR = ROOT / "outputs" / "figures" / "architecture_ablation_longonly"
BASELINE_PREDICTIONS = ROOT / "outputs" / "final" / "model_suite_top5" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"

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

TARGET_COLUMNS = [
    "target_ret_1d",
    "target_ret_3d",
    "target_ret_5d",
    "target_ret_10d",
    "target_excess_ret_5d",
    "target_rank_5d",
    "target_dir_5d",
]


@dataclass
class MultiTargetSet:
    X: np.ndarray
    y: dict[str, np.ndarray]
    meta: pd.DataFrame

    @property
    def size(self) -> int:
        return int(len(self.meta))


def main() -> None:
    set_seed(SEED)
    clean_outputs()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PREDICTION_DIR)
    ensure_dir(FIGURE_DIR)

    panel = prepare_panel()
    splits = build_multitarget_splits(panel)
    train_set = cap_multitarget_set(splits["train"], TRAIN_SAMPLE_CAP, SEED)
    valid_set = cap_multitarget_set(splits["valid"], VALID_SAMPLE_CAP, SEED)
    test_set = splits["test"]

    model = MultiHeadXLSTMLongOnlyRanker(num_features=len(FEATURE_COLUMNS), hidden_dim=96, num_blocks=2, dropout=0.1)
    history = train_model(model, train_set, valid_set)
    multihead_predictions = predict_model(model, test_set)
    baseline_predictions = load_baseline_for_same_rows(test_set)

    metrics = pd.DataFrame(
        [
            compute_all_metrics(baseline_predictions, model_label="Hybrid baseline", score_column="y_pred"),
            compute_all_metrics(multihead_predictions, model_label="A1 Multi-Head LongOnlyRank", score_column="y_pred"),
        ]
    )
    metrics["promote"] = False
    metrics.loc[metrics["model"].eq("A1 Multi-Head LongOnlyRank"), "promote"] = should_promote(metrics)

    save_table(pd.DataFrame(history), OUTPUT_DIR / "training_history.csv")
    save_table(metrics, OUTPUT_DIR / "architecture_ablation_metrics.csv")
    save_table(baseline_predictions, PREDICTION_DIR / "baseline_same_rows_predictions.parquet")
    save_table(multihead_predictions, PREDICTION_DIR / "a1_multihead_longonlyrank_predictions.parquet")
    write_report(metrics, history)
    plot_metrics(metrics)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


class MultiHeadXLSTMLongOnlyRanker(nn.Module):
    def __init__(self, *, num_features: int, hidden_dim: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualLSTMBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head_ret_1d = head(hidden_dim, dropout)
        self.head_ret_3d = head(hidden_dim, dropout)
        self.head_ret_5d = head(hidden_dim, dropout)
        self.head_ret_10d = head(hidden_dim, dropout)
        self.head_excess_5d = head(hidden_dim, dropout)
        self.head_rank_5d = head(hidden_dim, dropout)
        self.head_dir_5d = head(hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.input_projection(x)
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.output_norm(hidden[:, -1, :])
        return {
            "ret_1d": self.head_ret_1d(pooled).squeeze(-1),
            "ret_3d": self.head_ret_3d(pooled).squeeze(-1),
            "ret_5d": self.head_ret_5d(pooled).squeeze(-1),
            "ret_10d": self.head_ret_10d(pooled).squeeze(-1),
            "excess_5d": self.head_excess_5d(pooled).squeeze(-1),
            "rank_5d": self.head_rank_5d(pooled).squeeze(-1),
            "dir_5d": self.head_dir_5d(pooled).squeeze(-1),
        }


def head(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def prepare_panel() -> pd.DataFrame:
    panel = load_shared_feature_panel().sort_values(["symbol", "date"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_ret_10d"] = panel.groupby("symbol")["close"].shift(-10) / panel["close"] - 1.0
    panel["target_rank_5d"] = panel.groupby("date")["target_ret_5d"].rank(pct=True).astype(float) * 2.0 - 1.0
    return panel


def build_multitarget_splits(panel: pd.DataFrame) -> dict[str, MultiTargetSet]:
    base_splits, _ = build_scaled_sequence_splits(
        feature_panel=panel,
        feature_columns=FEATURE_COLUMNS,
        target_column="target_ret_5d",
        lookback=SEQ_LEN,
    )
    target_frame = panel[["symbol", "date", *TARGET_COLUMNS]].copy()
    output: dict[str, MultiTargetSet] = {}
    for split_name, base_set in base_splits.items():
        meta = base_set.meta.merge(target_frame, on=["symbol", "date"], how="left")
        valid_mask = meta[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        clean_meta = meta.loc[valid_mask].reset_index(drop=True)
        targets = {
            "ret_1d": clean_meta["target_ret_1d"].to_numpy(dtype=np.float32),
            "ret_3d": clean_meta["target_ret_3d"].to_numpy(dtype=np.float32),
            "ret_5d": clean_meta["target_ret_5d"].to_numpy(dtype=np.float32),
            "ret_10d": clean_meta["target_ret_10d"].to_numpy(dtype=np.float32),
            "excess_5d": clean_meta["target_excess_ret_5d"].to_numpy(dtype=np.float32),
            "rank_5d": clean_meta["target_rank_5d"].to_numpy(dtype=np.float32),
            "dir_5d": clean_meta["target_dir_5d"].to_numpy(dtype=np.float32),
        }
        output[split_name] = MultiTargetSet(X=base_set.X[valid_mask], y=targets, meta=clean_meta[["symbol", "date", "split"]])
    return output


def cap_multitarget_set(sample_set: MultiTargetSet, max_samples: int, seed: int) -> MultiTargetSet:
    if sample_set.size <= max_samples:
        return sample_set
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(sample_set.size, size=max_samples, replace=False))
    return MultiTargetSet(
        X=sample_set.X[indices],
        y={key: value[indices] for key, value in sample_set.y.items()},
        meta=sample_set.meta.iloc[indices].reset_index(drop=True),
    )


def train_model(model: nn.Module, train_set: MultiTargetSet, valid_set: MultiTargetSet) -> list[dict[str, float]]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    huber = nn.HuberLoss(delta=0.05)
    bce = nn.BCEWithLogitsLoss()
    best_score = -math.inf
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float]] = []
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        losses: list[float] = []
        for batch in loader_for(train_set, batch_size=512, shuffle=True):
            batch_x, batch_targets = unpack_batch(batch)
            batch_x = batch_x.to(DEVICE)
            batch_targets = {key: value.to(DEVICE) for key, value in batch_targets.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = multihead_loss(output, batch_targets, huber=huber, bce=bce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_prediction = predict_model(model, valid_set)
        valid_metrics = compute_all_metrics(valid_prediction, model_label="valid", score_column="y_pred")
        valid_score = validation_selection_score(valid_metrics)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "valid_selection_score": valid_score,
            "valid_IC": float(valid_metrics["IC"]),
            "valid_RankIC": float(valid_metrics["RankIC"]),
            "valid_Top5_Return": float(valid_metrics["Top5_Return"]),
            "valid_Top5_Direction_Acc": float(valid_metrics["Top5_Direction_Acc"]),
            "valid_total_return": float(valid_metrics["total_return"]),
            "seconds": time.time() - start,
        }
        history.append(row)
        if valid_score > best_score:
            best_score = valid_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 2:
                break
    model.load_state_dict(best_state)
    return history


def multihead_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    huber: nn.Module,
    bce: nn.Module,
) -> torch.Tensor:
    return (
        1.00 * huber(output["ret_5d"], target["ret_5d"])
        + 0.30 * huber(output["excess_5d"], target["excess_5d"])
        + 0.20 * huber(output["rank_5d"], target["rank_5d"])
        + 0.10 * bce(output["dir_5d"], target["dir_5d"])
        + 0.10 * huber(output["ret_3d"], target["ret_3d"])
        + 0.05 * huber(output["ret_10d"], target["ret_10d"])
        + 0.03 * huber(output["ret_1d"], target["ret_1d"])
    )


def loader_for(sample_set: MultiTargetSet, *, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [
        torch.tensor(sample_set.X, dtype=torch.float32),
        torch.tensor(sample_set.y["ret_1d"], dtype=torch.float32),
        torch.tensor(sample_set.y["ret_3d"], dtype=torch.float32),
        torch.tensor(sample_set.y["ret_5d"], dtype=torch.float32),
        torch.tensor(sample_set.y["ret_10d"], dtype=torch.float32),
        torch.tensor(sample_set.y["excess_5d"], dtype=torch.float32),
        torch.tensor(sample_set.y["rank_5d"], dtype=torch.float32),
        torch.tensor(sample_set.y["dir_5d"], dtype=torch.float32),
    ]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def unpack_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return batch[0], {
        "ret_1d": batch[1],
        "ret_3d": batch[2],
        "ret_5d": batch[3],
        "ret_10d": batch[4],
        "excess_5d": batch[5],
        "rank_5d": batch[6],
        "dir_5d": batch[7],
    }


def predict_model(model: nn.Module, sample_set: MultiTargetSet) -> pd.DataFrame:
    model.eval()
    outputs: list[pd.DataFrame] = []
    with torch.no_grad():
        start = 0
        for batch in loader_for(sample_set, batch_size=1024, shuffle=False):
            batch_x, _ = unpack_batch(batch)
            out = model(batch_x.to(DEVICE))
            batch_size = len(batch_x)
            meta = sample_set.meta.iloc[start : start + batch_size].copy().reset_index(drop=True)
            for key, value in out.items():
                meta[f"pred_{key}"] = value.detach().cpu().numpy().astype(np.float32, copy=False)
            meta["y_true"] = sample_set.y["ret_5d"][start : start + batch_size]
            outputs.append(meta)
            start += batch_size
    frame = pd.concat(outputs, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["y_pred"] = build_final_score(frame)
    context = PredictionContext(
        model_family="A1 Multi-Head LongOnlyRank",
        model_version="a1_multihead_longonlyrank_v1",
        target_name="target_ret_5d",
        horizon=5,
        run_id="a1_multihead_longonlyrank_v1",
    )
    return standardize_prediction_output(frame, context, y_true_column="y_true")


def build_final_score(frame: pd.DataFrame) -> np.ndarray:
    parts = {
        "rank": 0.50 * grouped_zscore(frame, "pred_rank_5d"),
        "ret5": 0.25 * grouped_zscore(frame, "pred_ret_5d"),
        "excess": 0.15 * grouped_zscore(frame, "pred_excess_5d"),
        "dir": 0.10 * grouped_zscore(frame, "pred_dir_5d"),
    }
    score = sum(parts.values())
    return score.to_numpy(dtype=np.float32)


def grouped_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(float)
    return values.groupby(frame["date"]).transform(lambda group: (group - group.mean()) / group.std(ddof=0) if group.std(ddof=0) > 1e-12 else group * 0.0)


def load_baseline_for_same_rows(test_set: MultiTargetSet) -> pd.DataFrame:
    baseline = load_table(BASELINE_PREDICTIONS).copy()
    baseline["date"] = pd.to_datetime(baseline["date"])
    keys = test_set.meta[["symbol", "date"]].copy()
    output = keys.merge(
        baseline[["model_family", "model_version", "symbol", "date", "split", "y_true", "y_pred", "target_name", "horizon", "run_id"]],
        on=["symbol", "date"],
        how="inner",
    )
    output["model_family"] = "Hybrid xLSTM Direction-Excess Blend"
    output["model_version"] = "baseline_same_rows"
    output["run_id"] = "baseline_same_rows"
    return output


def compute_all_metrics(frame: pd.DataFrame, *, model_label: str, score_column: str) -> dict[str, object]:
    test = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if test.empty:
        test = frame.copy()
    test["date"] = pd.to_datetime(test["date"])
    cross_section = compute_cross_section_stats(test, score_column=score_column, top_k=TOP_K)
    y_true = test["y_true"].to_numpy(dtype=float)
    y_score = test[score_column].to_numpy(dtype=float)
    threshold = tune_threshold(y_true, y_score)
    backtest = run_backtest(
        test,
        mode="long-only",
        score_column=score_column,
        top_k=TOP_K,
        rebalance_every=5,
        initial_capital=100_000_000.0,
        transaction_cost_bps=15.0,
        cross_section=cross_section,
    ).summary
    return {
        "model": model_label,
        "rows": int(len(test)),
        "dates": int(test["date"].nunique()),
        "symbols": int(test["symbol"].nunique()),
        "IC": cross_section["IC"],
        "RankIC": cross_section["RankIC"],
        "ICIR": cross_section["ICIR"],
        "RankICIR": cross_section["RankICIR"],
        "Direction_Acc": directional_accuracy(y_true, y_score, threshold),
        "Balanced_Acc": balanced_directional_accuracy(y_true, y_score, threshold),
        "Majority": majority_baseline_accuracy(y_true),
        "F1_up": f1_up(y_true, y_score, threshold),
        "F1_down": f1_down(y_true, y_score, threshold),
        "Top5_Return": cross_section["TopK_Return"],
        "Bottom5_Return": cross_section["BottomK_Return"],
        "LongShort5_Diagnostic": cross_section["LongShort"],
        "Top5_Direction_Acc": cross_section["TopK_Direction_Acc"],
        "total_return": backtest["total_return"],
        "final_capital": backtest["final_capital"],
        "sharpe_proxy": backtest["sharpe_proxy"],
        "max_drawdown": backtest["max_drawdown"],
        "avg_period_return": backtest["avg_period_return"],
    }


def validation_selection_score(metrics: dict[str, object]) -> float:
    return (
        0.35 * float(metrics["Top5_Return"])
        + 0.25 * float(metrics["Top5_Direction_Acc"])
        + 0.20 * float(metrics["RankIC"])
        + 0.10 * float(metrics["IC"])
        + 0.10 * float(metrics["total_return"])
    )


def should_promote(metrics: pd.DataFrame) -> bool:
    baseline = metrics.loc[metrics["model"].eq("Hybrid baseline")].iloc[0]
    candidate = metrics.loc[metrics["model"].eq("A1 Multi-Head LongOnlyRank")].iloc[0]
    return bool(
        candidate["total_return"] >= baseline["total_return"]
        and candidate["Top5_Return"] >= baseline["Top5_Return"]
        and candidate["Top5_Direction_Acc"] >= baseline["Top5_Direction_Acc"] - 0.002
        and candidate["RankIC"] >= baseline["RankIC"] - 0.005
    )


def tune_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    finite = score[np.isfinite(score)]
    if len(finite) == 0:
        return 0.0
    candidates = np.unique(np.concatenate([np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]), np.quantile(finite, [0.35, 0.5, 0.65])]))
    best_threshold = 0.0
    best_value = -math.inf
    for threshold in candidates:
        value = balanced_directional_accuracy(y_true, score, float(threshold))
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def write_report(metrics: pd.DataFrame, history: list[dict[str, float]]) -> None:
    metric_cols = [
        "model",
        "rows",
        "IC",
        "RankIC",
        "Top5_Return",
        "Top5_Direction_Acc",
        "total_return",
        "final_capital",
        "sharpe_proxy",
        "max_drawdown",
        "promote",
    ]
    lines = [
        "# Architecture Ablation: A1 Multi-Head LongOnlyRank",
        "",
        "Goal: test an architecture-only upgrade while keeping the Hybrid xLSTM baseline as the production reference.",
        "",
        "A1 changes:",
        "",
        "- Same residual xLSTM-style backbone size: hidden `96`, `2` blocks, dropout `0.1`.",
        "- Multi-head outputs: `ret_1d`, `ret_3d`, `ret_5d`, computed `ret_10d`, `excess_5d`, `rank_5d`, `dir_5d`.",
        "- Final long-only score: `0.50*z(rank_5d) + 0.25*z(ret_5d) + 0.15*z(excess_5d) + 0.10*z(dir_5d)`.",
        "- Checkpointing uses validation long-only score, not validation loss.",
        "",
        "Promotion rule:",
        "",
        "- `total_return >= baseline`",
        "- `Top5_Return >= baseline`",
        "- `Top5_Direction_Acc` not worse by more than `0.2pp`",
        "- `RankIC` not worse by more than `0.005`",
        "",
        "## Locked Test Metrics",
        "",
        _markdown_table(metrics[metric_cols]),
        "",
        "## Training History",
        "",
        _markdown_table(pd.DataFrame(history)),
        "",
        "## Decision",
        "",
        "Promote A1 only if the `promote` column is true. Otherwise keep `Hybrid xLSTM Direction-Excess Blend` as production.",
        "",
        "Artifacts:",
        "",
        "- `outputs/reports/architecture_ablation_longonly/architecture_ablation_metrics.csv`",
        "- `outputs/reports/architecture_ablation_longonly/training_history.csv`",
        "- `outputs/final/architecture_ablation_longonly/a1_multihead_longonlyrank_predictions.parquet`",
        "- `outputs/final/architecture_ablation_longonly/baseline_same_rows_predictions.parquet`",
        "- `outputs/figures/architecture_ablation_longonly/test_total_return.png`",
    ]
    (OUTPUT_DIR / "architecture_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(metrics["model"], metrics["total_return"])
    plt.title("Architecture ablation: long-only test total return")
    plt.ylabel("total return after cost")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "test_total_return.png", dpi=160)
    plt.close()


def clean_outputs() -> None:
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
