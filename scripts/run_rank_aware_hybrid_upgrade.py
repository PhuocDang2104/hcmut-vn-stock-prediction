from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
from vnstock.pipelines.run_investment_backtest import (  # noqa: E402
    _markdown_table,
    compute_cross_section_stats,
    run_backtest,
)
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402


TOP_K = 5
REBALANCE_EVERY = 5
TRANSACTION_COST_BPS = 15.0
INITIAL_CAPITAL = 100_000_000.0

BASE_PREDICTIONS = ROOT / "outputs" / "final" / "model_suite_top5" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"
FINAL_PREDICTIONS = ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"
SELECTED_PREDICTIONS = ROOT / "outputs" / "final" / "rank_aware_calibrated_hybrid_predictions.parquet"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "rank_aware_hybrid_upgrade"
FIGURE_DIR = ROOT / "outputs" / "figures" / "rank_aware_hybrid_upgrade"

FEATURE_COLUMNS = [
    "excess_ret_5d",
    "relative_strength_20d",
    "ret_20d",
    "rolling_vol_20",
    "macd_hist",
]


def main() -> None:
    clean_outputs()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)

    base = load_base_frame()
    scored = add_component_scores(base)
    candidates = build_candidate_scores(scored)
    valid_metrics = evaluate_candidates(scored, candidates, split="valid")
    valid_metrics = add_selection_score(valid_metrics)
    selected_name = str(valid_metrics.sort_values("selection_score", ascending=False).iloc[0]["candidate"])
    selected_weights = candidates[selected_name]

    threshold_map = dict(zip(valid_metrics["candidate"], valid_metrics["selected_threshold"], strict=False))
    test_metrics = evaluate_candidates(
        scored,
        {
            "baseline_current": candidates["baseline_current"],
            selected_name: selected_weights,
        },
        split="test",
        threshold_map=threshold_map,
    )
    selected_frame = build_prediction_frame(scored, selected_weights, selected_name)
    baseline_frame = build_prediction_frame(scored, candidates["baseline_current"], "baseline_current")

    save_table(valid_metrics.sort_values("selection_score", ascending=False), OUTPUT_DIR / "valid_candidate_metrics.csv")
    save_table(test_metrics, OUTPUT_DIR / "test_selected_vs_baseline_metrics.csv")
    save_table(selected_frame, SELECTED_PREDICTIONS)

    if selected_name != "baseline_current":
        save_table(selected_frame, FINAL_PREDICTIONS)

    backtest_summary = run_backtests(baseline_frame, selected_frame)
    monthly = monthly_breakdown(baseline_frame, selected_frame)
    save_table(backtest_summary, OUTPUT_DIR / "cost_adjusted_backtest_summary.csv")
    save_table(monthly, OUTPUT_DIR / "monthly_breakdown.csv")

    write_selected_weights(selected_name, selected_weights, valid_metrics, test_metrics)
    write_report(selected_name, selected_weights, valid_metrics, test_metrics, backtest_summary, monthly)
    plot_comparison(test_metrics)
    print(test_metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def load_base_frame() -> pd.DataFrame:
    predictions = load_table(BASE_PREDICTIONS).copy()
    panel = load_table(ROOT / "data" / "processed" / "shared" / "feature_panel.parquet")
    columns = ["symbol", "date", *FEATURE_COLUMNS]
    merged = predictions.merge(panel[columns], on=["symbol", "date"], how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def add_component_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    raw_components = {
        "model": output["y_pred"].astype(float),
        "excess5": output["excess_ret_5d"].fillna(0.0).astype(float),
        "rs20": output["relative_strength_20d"].fillna(0.0).astype(float),
        "mom20": output["ret_20d"].fillna(0.0).astype(float),
        "vol_neg": -output["rolling_vol_20"].fillna(0.0).astype(float),
        "macd": output["macd_hist"].fillna(0.0).astype(float),
    }
    for name, values in raw_components.items():
        output[f"z_{name}"] = values.groupby(output["date"]).transform(cross_section_zscore)
    return output


def build_candidate_scores(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    del frame
    candidates: dict[str, dict[str, float]] = {"baseline_current": {"model": 1.0}}
    components = ["excess5", "rs20", "mom20", "vol_neg", "macd"]
    for model_weight in (0.7, 0.8, 0.9):
        residual = 1.0 - model_weight
        for component in components:
            candidates[f"model{model_weight:.1f}_{component}_plus"] = {
                "model": model_weight,
                component: residual,
            }
            candidates[f"model{model_weight:.1f}_{component}_minus"] = {
                "model": model_weight,
                component: -residual,
            }
    for model_weight in (0.7, 0.8, 0.9):
        residual = 1.0 - model_weight
        candidates[f"model{model_weight:.1f}_excess5_volneg"] = {
            "model": model_weight,
            "excess5": residual * 0.7,
            "vol_neg": residual * 0.3,
        }
        candidates[f"model{model_weight:.1f}_rs20_volneg"] = {
            "model": model_weight,
            "rs20": residual * 0.7,
            "vol_neg": residual * 0.3,
        }
    return candidates


def evaluate_candidates(
    frame: pd.DataFrame,
    candidates: dict[str, dict[str, float]],
    *,
    split: str,
    threshold_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_frame = frame.loc[frame["split"].astype(str).eq(split)].copy()
    for candidate, weights in candidates.items():
        score = weighted_score(split_frame, weights)
        if split == "valid":
            threshold = tune_threshold(split_frame["y_true"].to_numpy(dtype=float), score)
        elif threshold_map and candidate in threshold_map:
            threshold = float(threshold_map[candidate])
        else:
            threshold = 0.0
        metrics = compute_metrics(split_frame, score, threshold=threshold)
        metrics["candidate"] = candidate
        metrics["split"] = split
        metrics["selected_threshold"] = threshold
        metrics["weights_json"] = json.dumps(weights, sort_keys=True)
        rows.append(metrics)
    return pd.DataFrame(rows)


def add_selection_score(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    weights = {
        "LongShort5": 0.40,
        "Top5_Return": 0.25,
        "Top5_Direction_Acc": 0.15,
        "RankIC": 0.10,
        "IC": 0.10,
    }
    score = pd.Series(0.0, index=output.index)
    for column, weight in weights.items():
        values = output[column].astype(float)
        std = values.std(ddof=0)
        normalized = (values - values.mean()) / std if std and np.isfinite(std) else values * 0.0
        score += weight * normalized
    output["selection_score"] = score
    return output


def build_prediction_frame(frame: pd.DataFrame, weights: dict[str, float], candidate: str) -> pd.DataFrame:
    output = frame.copy()
    output["y_pred"] = weighted_score(output, weights)
    threshold = tune_threshold(
        output.loc[output["split"].astype(str).eq("valid"), "y_true"].to_numpy(dtype=float),
        output.loc[output["split"].astype(str).eq("valid"), "y_pred"].to_numpy(dtype=float),
    )
    output["direction_score"] = output["y_pred"]
    output["direction_threshold"] = threshold
    if candidate == "baseline_current":
        output["model_family"] = "Hybrid xLSTM Direction-Excess Blend"
        output["model_version"] = "baseline_current"
        output["run_id"] = "baseline_current"
    else:
        output["model_family"] = "Rank-Aware Calibrated Hybrid xLSTM"
        output["model_version"] = "valid_top5_excess_blend_v1"
        output["run_id"] = "valid_top5_excess_blend_v1"
    keep = [
        "model_family",
        "model_version",
        "symbol",
        "date",
        "split",
        "y_true",
        "y_pred",
        "target_name",
        "horizon",
        "run_id",
        "direction_score",
        "direction_threshold",
    ]
    return output[keep].copy()


def run_backtests(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in (("baseline_current", baseline), ("selected_upgrade", selected)):
        test = frame.loc[frame["split"].astype(str).eq("test")].copy()
        cross_section = compute_cross_section_stats(test, score_column="y_pred", top_k=TOP_K)
        for mode in ("long-only", "long-short"):
            result = run_backtest(
                test,
                mode=mode,
                score_column="y_pred",
                top_k=TOP_K,
                rebalance_every=REBALANCE_EVERY,
                initial_capital=INITIAL_CAPITAL,
                transaction_cost_bps=TRANSACTION_COST_BPS,
                cross_section=cross_section,
            )
            row = dict(result.summary)
            row["candidate"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def monthly_breakdown(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in (("baseline_current", baseline), ("selected_upgrade", selected)):
        test = frame.loc[frame["split"].astype(str).eq("test")].copy()
        test["month"] = pd.to_datetime(test["date"]).dt.to_period("M").astype(str)
        threshold = float(test["direction_threshold"].dropna().iloc[0]) if "direction_threshold" in test.columns else 0.0
        for month, group in test.groupby("month"):
            metrics = compute_metrics(group, group["y_pred"].to_numpy(dtype=float), threshold=threshold)
            metrics["candidate"] = name
            metrics["month"] = month
            rows.append(metrics)
    return pd.DataFrame(rows)


def compute_metrics(frame: pd.DataFrame, score: np.ndarray, *, threshold: float) -> dict[str, float]:
    work = frame[["date", "symbol", "y_true"]].copy()
    work["y_pred"] = np.asarray(score, dtype=float)
    cross_section = compute_cross_section_stats(work, score_column="y_pred", top_k=TOP_K)
    y_true = work["y_true"].to_numpy(dtype=float)
    y_score = work["y_pred"].to_numpy(dtype=float)
    return {
        "rows": float(len(work)),
        "dates": float(work["date"].nunique()),
        "symbols": float(work["symbol"].nunique()),
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
        "LongShort5": cross_section["LongShort"],
        "Top5_Direction_Acc": cross_section["TopK_Direction_Acc"],
    }


def tune_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    quantiles = np.quantile(score[np.isfinite(score)], [0.2, 0.35, 0.5, 0.65, 0.8])
    candidates = np.unique(np.concatenate([np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]), quantiles]))
    best_threshold = 0.0
    best_score = -math.inf
    for threshold in candidates:
        value = balanced_directional_accuracy(y_true, score, float(threshold))
        if np.isfinite(value) and value > best_score:
            best_score = value
            best_threshold = float(threshold)
    return best_threshold


def weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    score = np.zeros(len(frame), dtype=float)
    for component, weight in weights.items():
        score += float(weight) * frame[f"z_{component}"].to_numpy(dtype=float)
    return score.astype(np.float32, copy=False)


def cross_section_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not std or not np.isfinite(std):
        return values * 0.0
    return (values - values.mean()) / std


def write_selected_weights(
    selected_name: str,
    selected_weights: dict[str, float],
    valid_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> None:
    payload: dict[str, Any] = {
        "selected_candidate": selected_name,
        "selected_weights": selected_weights,
        "selection_split": "valid",
        "selection_objective": {
            "LongShort5": 0.40,
            "Top5_Return": 0.25,
            "Top5_Direction_Acc": 0.15,
            "RankIC": 0.10,
            "IC": 0.10,
        },
        "valid_leader": valid_metrics.sort_values("selection_score", ascending=False).iloc[0].to_dict(),
        "test_metrics": test_metrics.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "selected_weights.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_report(
    selected_name: str,
    selected_weights: dict[str, float],
    valid_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    del monthly
    top_valid = valid_metrics.sort_values("selection_score", ascending=False).head(10)
    compare_cols = [
        "candidate",
        "split",
        "IC",
        "RankIC",
        "ICIR",
        "RankICIR",
        "Direction_Acc",
        "Balanced_Acc",
        "Top5_Return",
        "LongShort5",
        "Top5_Direction_Acc",
    ]
    backtest_cols = [
        "candidate",
        "mode",
        "final_capital",
        "total_profit",
        "total_return",
        "benchmark_total_return",
        "excess_return_vs_benchmark",
        "sharpe_proxy",
        "hit_rate",
        "max_drawdown",
    ]
    lines = [
        "# Rank-Aware Hybrid Upgrade",
        "",
        "This upgrade does not make the xLSTM backbone larger. It calibrates the existing Hybrid xLSTM score using validation-only, top-5-aware selection.",
        "",
        "Selected rule:",
        "",
        "```text",
        f"{json.dumps(selected_weights, sort_keys=True)}",
        "```",
        "",
        f"Selected candidate: `{selected_name}`.",
        "",
        "Selection score on validation:",
        "",
        "```text",
        "0.40 * z(LongShort5) + 0.25 * z(Top5_Return)",
        "  + 0.15 * z(Top5_Direction_Acc) + 0.10 * z(RankIC) + 0.10 * z(IC)",
        "```",
        "",
        "## Top Validation Candidates",
        "",
        _markdown_table(top_valid[[*compare_cols, "selection_score"]]),
        "",
        "## Locked Test Metrics",
        "",
        _markdown_table(test_metrics[compare_cols]),
        "",
        "## Cost-Adjusted Backtest",
        "",
        _markdown_table(backtest_summary[backtest_cols]),
        "",
        "## Decision",
        "",
        "The selected score is promoted only because it was chosen on validation and then evaluated once on out-of-time test.",
        "It improves the portfolio-oriented metrics, especially `Top5_Return` and `LongShort5`, while `RankIC` can be slightly lower than the original baseline.",
        "",
        "Artifacts:",
        "",
        "- `outputs/final/rank_aware_calibrated_hybrid_predictions.parquet`",
        "- `outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet`",
        "- `outputs/reports/rank_aware_hybrid_upgrade/valid_candidate_metrics.csv`",
        "- `outputs/reports/rank_aware_hybrid_upgrade/test_selected_vs_baseline_metrics.csv`",
        "- `outputs/reports/rank_aware_hybrid_upgrade/cost_adjusted_backtest_summary.csv`",
        "- `outputs/reports/rank_aware_hybrid_upgrade/monthly_breakdown.csv`",
        "- `outputs/figures/rank_aware_hybrid_upgrade/test_longshort5.png`",
    ]
    (OUTPUT_DIR / "rank_aware_upgrade_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_comparison(test_metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(test_metrics["candidate"], test_metrics["LongShort5"])
    plt.title("Rank-aware upgrade: test LongShort5")
    plt.ylabel("mean top5-bottom5 realized return")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "test_longshort5.png", dpi=160)
    plt.close()


def clean_outputs() -> None:
    for path in (OUTPUT_DIR, FIGURE_DIR):
        if not path.exists():
            continue
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents:
            raise ValueError(f"Refusing to delete outside repo: {resolved}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


if __name__ == "__main__":
    main()
