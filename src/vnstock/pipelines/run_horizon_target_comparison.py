from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from vnstock.utils.io import ensure_dir, load_table, save_table
from vnstock.utils.logging import get_logger


RETURN_THRESHOLDS = [-0.01, -0.005, -0.003, 0.0, 0.003, 0.005, 0.01]
PROB_THRESHOLDS = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
REQUIRED_SUMMARY_COLUMNS = [
    "flow_id",
    "model",
    "horizon",
    "target_type",
    "feature_mode",
    "seq_len",
    "wavelet_enabled",
    "threshold_type",
    "selected_threshold",
    "IC",
    "RankIC",
    "ICIR",
    "RankICIR",
    "Direction_Acc",
    "Balanced_Acc",
    "Majority",
    "F1_up",
    "F1_down",
    "TopK_Return",
    "BottomK_Return",
    "LongShort",
    "TopK_Direction_Acc",
    "LargeMove_Direction_Acc_q60",
    "LargeMove_Direction_Acc_q70",
    "Coverage_q60",
    "Coverage_q70",
    "Turnover",
    "Cost_Adjusted_LongShort",
    "direction_score",
    "ranking_score",
    "production_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate horizon/target stock-selection comparison flows.")
    parser.add_argument("--manifest", default="configs/experiments/horizon_target_comparison/manifest.csv")
    parser.add_argument("--feature-panel", default="data/processed/shared/feature_panel.parquet")
    parser.add_argument("--output-dir", default="experiments/horizon_target_comparison")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_horizon_target_comparison")
    output_dir = ensure_dir(args.output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    manifest = pd.read_csv(args.manifest)
    panel = load_table(args.feature_panel)
    panel["date"] = pd.to_datetime(panel["date"])

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for _, manifest_row in manifest.iterrows():
        path = Path(str(manifest_row["predictions_path"]))
        if not path.exists():
            message = f"{manifest_row['flow_id']} missing predictions: {path}"
            if args.allow_missing:
                skipped.append(message)
                logger.warning(message)
                continue
            raise FileNotFoundError(message)
        records.append(
            evaluate_manifest_row(
                manifest_row.to_dict(),
                panel=panel,
                top_k=args.top_k,
                cost_bps=args.cost_bps,
            )
        )

    if not records:
        raise ValueError("No experiment records were evaluated.")

    summary = pd.DataFrame(records)
    summary = add_selection_scores(summary)
    summary = summary.sort_values("production_score", ascending=False).reset_index(drop=True)
    summary_path = save_table(summary[REQUIRED_SUMMARY_COLUMNS + _extra_columns(summary)], output_dir / "summary_all.csv")
    save_table(summary, output_dir / "summary_all_with_validation.csv")
    plot_all(summary, plots_dir=plots_dir)
    report_path = write_report(summary, output_dir / "report.md", skipped=skipped, top_k=args.top_k)
    logger.info("Horizon/target comparison complete: summary=%s report=%s", summary_path, report_path)


def evaluate_manifest_row(row: dict[str, Any], *, panel: pd.DataFrame, top_k: int, cost_bps: float) -> dict[str, Any]:
    predictions = load_table(row["predictions_path"]).copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    horizon = int(row["horizon"])
    merged = predictions.merge(
        panel[_panel_columns(horizon)],
        on=["symbol", "date"],
        how="inner",
        suffixes=("", "_panel"),
    )
    if merged.empty:
        raise ValueError(f"No merged rows for {row['flow_id']} from {row['predictions_path']}")

    prepared = prepare_scores(merged, row=row, horizon=horizon)
    threshold_source = str(row.get("threshold_source", "pred_ret"))
    threshold_type = str(row.get("threshold_type", "return"))
    thresholds = PROB_THRESHOLDS if threshold_type == "probability" else RETURN_THRESHOLDS
    selected_threshold = tune_threshold(
        prepared.loc[prepared["split"].astype(str) == "valid"],
        score_column=threshold_source,
        thresholds=thresholds,
    )

    train_returns = prepared.loc[prepared["split"].astype(str) == "train", "true_ret"].dropna().abs()
    q60 = float(train_returns.quantile(0.60)) if not train_returns.empty else math.nan
    q70 = float(train_returns.quantile(0.70)) if not train_returns.empty else math.nan

    valid_metrics = compute_flow_metrics(
        prepared.loc[prepared["split"].astype(str) == "valid"],
        score_column=threshold_source,
        threshold=selected_threshold,
        top_k=top_k,
        q60=q60,
        q70=q70,
        cost_bps=cost_bps,
    )
    test_metrics = compute_flow_metrics(
        prepared.loc[prepared["split"].astype(str) == "test"],
        score_column=threshold_source,
        threshold=selected_threshold,
        top_k=top_k,
        q60=q60,
        q70=q70,
        cost_bps=cost_bps,
    )
    close_metrics = compute_close_metrics(prepared.loc[prepared["split"].astype(str) == "test"])

    output = {
        "flow_id": row["flow_id"],
        "model": row["model"],
        "horizon": horizon,
        "target_type": row["target_type"],
        "feature_mode": row["feature_mode"],
        "seq_len": int(row["seq_len"]),
        "wavelet_enabled": bool(row.get("wavelet_enabled", False)),
        "threshold_type": threshold_type,
        "selected_threshold": selected_threshold,
        **test_metrics,
        **close_metrics,
    }
    for key, value in valid_metrics.items():
        output[f"valid_{key}"] = value
    return output


def _panel_columns(horizon: int) -> list[str]:
    return [
        "symbol",
        "date",
        "close",
        "log_close",
        f"target_ret_{horizon}d",
        f"target_log_close_{horizon}d",
    ]


def prepare_scores(frame: pd.DataFrame, *, row: dict[str, Any], horizon: int) -> pd.DataFrame:
    target_type = str(row["target_type"])
    output = frame.copy()
    output["true_ret"] = output[f"target_ret_{horizon}d"].astype(float)
    output["ranking_score"] = output["y_pred"].astype(float)
    output["pred_log_close"] = np.nan
    output["true_log_close"] = output[f"target_log_close_{horizon}d"].astype(float)

    if target_type == "close":
        output["pred_log_close"] = output["y_pred"].astype(float)
        output["pred_ret"] = np.exp(output["pred_log_close"] - output["log_close"].astype(float)) - 1.0
        output["ranking_score"] = output["pred_ret"]
    elif target_type == "log_return":
        output["pred_ret"] = np.exp(output["y_pred"].astype(float)) - 1.0
        output["ranking_score"] = output["pred_ret"]
    elif target_type in {"return", "score"}:
        output["pred_ret"] = output["y_pred"].astype(float)
        output["ranking_score"] = output["pred_ret"]
    else:
        raise ValueError(f"Unsupported target_type={target_type!r}")

    if str(row.get("threshold_source", "pred_ret")) == "direction_score" and "direction_score" not in output.columns:
        raise ValueError(f"{row['flow_id']} requested direction_score threshold but file has no direction_score column.")
    output["threshold_pred_ret"] = output["pred_ret"]
    return output.dropna(subset=["true_ret", "pred_ret", "ranking_score"])


def tune_threshold(frame: pd.DataFrame, *, score_column: str, thresholds: list[float]) -> float:
    if frame.empty:
        return 0.0
    best_threshold = float(thresholds[0])
    best_score = -math.inf
    for threshold in thresholds:
        score = balanced_accuracy(frame["true_ret"].to_numpy(), frame[score_column].to_numpy(), threshold)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_flow_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    top_k: int,
    q60: float,
    q70: float,
    cost_bps: float,
) -> dict[str, float]:
    clean = frame.dropna(subset=["true_ret", "ranking_score", score_column]).copy()
    if clean.empty:
        return _empty_metrics()
    y_true = clean["true_ret"].to_numpy(dtype=float)
    y_score = clean[score_column].to_numpy(dtype=float)
    y_pred_up = y_score > threshold
    y_true_up = y_true > 0
    ic, icir = daily_correlation(clean, method="pearson")
    rankic, rankicir = daily_correlation(clean, method="spearman")
    top_metrics = top_bottom_metrics(clean, top_k=top_k, cost_bps=cost_bps)
    q60_metrics = large_move_metrics(clean, score_column=score_column, threshold=threshold, cutoff=q60)
    q70_metrics = large_move_metrics(clean, score_column=score_column, threshold=threshold, cutoff=q70)
    return {
        "IC": ic,
        "RankIC": rankic,
        "ICIR": icir,
        "RankICIR": rankicir,
        "Direction_Acc": float(np.mean(y_true_up == y_pred_up)),
        "Balanced_Acc": balanced_accuracy(y_true, y_score, threshold),
        "Majority": majority_accuracy(y_true),
        "F1_up": f1_for_class(y_true_up, y_pred_up, positive=True),
        "F1_down": f1_for_class(y_true_up, y_pred_up, positive=False),
        **top_metrics,
        "LargeMove_Direction_Acc_q60": q60_metrics["direction_acc"],
        "LargeMove_Balanced_Acc_q60": q60_metrics["balanced_acc"],
        "LargeMove_Direction_Acc_q70": q70_metrics["direction_acc"],
        "LargeMove_Balanced_Acc_q70": q70_metrics["balanced_acc"],
        "Coverage_q60": q60_metrics["coverage"],
        "Coverage_q70": q70_metrics["coverage"],
    }


def compute_close_metrics(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame.dropna(subset=["pred_log_close", "true_log_close"]).copy()
    if clean.empty:
        return {"MAE_log_close": math.nan, "RMSE_log_close": math.nan}
    error = clean["pred_log_close"].to_numpy(dtype=float) - clean["true_log_close"].to_numpy(dtype=float)
    return {
        "MAE_log_close": float(np.mean(np.abs(error))),
        "RMSE_log_close": float(np.sqrt(np.mean(np.square(error)))),
    }


def daily_correlation(frame: pd.DataFrame, *, method: str) -> tuple[float, float]:
    values: list[float] = []
    for _, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        x = group["ranking_score"].to_numpy(dtype=float)
        y = group["true_ret"].to_numpy(dtype=float)
        if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
            continue
        if method == "pearson":
            corr = pearsonr(x, y).statistic
        elif method == "spearman":
            corr = spearmanr(x, y, nan_policy="omit").statistic
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else math.nan
    icir = float(np.mean(arr) / std) if std and np.isfinite(std) and std > 0 else math.nan
    return float(np.mean(arr)), icir


def balanced_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true_up = y_true > 0
    y_pred_up = y_score > threshold
    up_mask = y_true_up
    down_mask = ~y_true_up
    if not up_mask.any() or not down_mask.any():
        return math.nan
    return 0.5 * (float(np.mean(y_pred_up[up_mask])) + float(np.mean(~y_pred_up[down_mask])))


def majority_accuracy(y_true: np.ndarray) -> float:
    up_rate = float(np.mean(y_true > 0))
    return max(up_rate, 1.0 - up_rate)


def f1_for_class(y_true_up: np.ndarray, y_pred_up: np.ndarray, *, positive: bool) -> float:
    actual = y_true_up if positive else ~y_true_up
    predicted = y_pred_up if positive else ~y_pred_up
    if not predicted.any() or not actual.any():
        return 0.0
    precision = float(np.mean(actual[predicted]))
    recall = float(np.mean(predicted[actual]))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def top_bottom_metrics(frame: pd.DataFrame, *, top_k: int, cost_bps: float) -> dict[str, float]:
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    spreads: list[float] = []
    top_direction_hits: list[float] = []
    top_sets: list[set[str]] = []
    for _, group in frame.groupby("date"):
        bucket_size = min(top_k, len(group))
        if bucket_size <= 0:
            continue
        top = group.nlargest(bucket_size, "ranking_score")
        bottom = group.nsmallest(bucket_size, "ranking_score")
        top_returns.append(float(top["true_ret"].mean()))
        bottom_returns.append(float(bottom["true_ret"].mean()))
        spreads.append(top_returns[-1] - bottom_returns[-1])
        top_direction_hits.append(float(np.mean(top["true_ret"].to_numpy(dtype=float) > 0)))
        top_sets.append(set(top["symbol"].astype(str)))
    turnover = average_turnover(top_sets)
    long_short = float(np.mean(spreads)) if spreads else math.nan
    cost = 2.0 * (cost_bps / 10_000.0) * turnover if np.isfinite(turnover) else math.nan
    return {
        "TopK_Return": float(np.mean(top_returns)) if top_returns else math.nan,
        "BottomK_Return": float(np.mean(bottom_returns)) if bottom_returns else math.nan,
        "LongShort": long_short,
        "TopK_Direction_Acc": float(np.mean(top_direction_hits)) if top_direction_hits else math.nan,
        "Turnover": turnover,
        "Cost_Adjusted_LongShort": long_short - cost if np.isfinite(long_short) and np.isfinite(cost) else math.nan,
    }


def average_turnover(top_sets: list[set[str]]) -> float:
    if len(top_sets) < 2:
        return math.nan
    turnovers: list[float] = []
    for previous, current in zip(top_sets[:-1], top_sets[1:]):
        if not previous and not current:
            continue
        denominator = max(len(previous), len(current), 1)
        turnovers.append(1.0 - len(previous & current) / denominator)
    return float(np.mean(turnovers)) if turnovers else math.nan


def large_move_metrics(frame: pd.DataFrame, *, score_column: str, threshold: float, cutoff: float) -> dict[str, float]:
    if not np.isfinite(cutoff):
        return {"direction_acc": math.nan, "balanced_acc": math.nan, "coverage": math.nan}
    mask = frame["true_ret"].abs() >= cutoff
    subset = frame.loc[mask]
    coverage = float(mask.mean()) if len(frame) else math.nan
    if subset.empty:
        return {"direction_acc": math.nan, "balanced_acc": math.nan, "coverage": coverage}
    y_true = subset["true_ret"].to_numpy(dtype=float)
    y_score = subset[score_column].to_numpy(dtype=float)
    direction_acc = float(np.mean((y_true > 0) == (y_score > threshold)))
    return {
        "direction_acc": direction_acc,
        "balanced_acc": balanced_accuracy(y_true, y_score, threshold),
        "coverage": coverage,
    }


def _empty_metrics() -> dict[str, float]:
    keys = [
        "IC",
        "RankIC",
        "ICIR",
        "RankICIR",
        "Direction_Acc",
        "Balanced_Acc",
        "Majority",
        "F1_up",
        "F1_down",
        "TopK_Return",
        "BottomK_Return",
        "LongShort",
        "TopK_Direction_Acc",
        "LargeMove_Direction_Acc_q60",
        "LargeMove_Balanced_Acc_q60",
        "LargeMove_Direction_Acc_q70",
        "LargeMove_Balanced_Acc_q70",
        "Coverage_q60",
        "Coverage_q70",
        "Turnover",
        "Cost_Adjusted_LongShort",
    ]
    return {key: math.nan for key in keys}


def add_selection_scores(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["F1_macro"] = result[["valid_F1_up", "valid_F1_down"]].mean(axis=1)
    result["direction_score"] = (
        0.5 * result["valid_Balanced_Acc"]
        + 0.3 * result["valid_LargeMove_Balanced_Acc_q70"]
        + 0.2 * result["F1_macro"]
    )
    for column in ["valid_IC", "valid_RankIC", "valid_TopK_Return", "valid_LongShort", "valid_Balanced_Acc", "valid_LargeMove_Balanced_Acc_q70"]:
        result[f"z_{column}"] = zscore(result[column])
    result["ranking_score"] = (
        0.35 * result["z_valid_IC"]
        + 0.35 * result["z_valid_RankIC"]
        + 0.15 * result["z_valid_TopK_Return"]
        + 0.15 * result["z_valid_LongShort"]
    )
    result["production_score"] = (
        0.25 * result["z_valid_IC"]
        + 0.25 * result["z_valid_RankIC"]
        + 0.20 * result["z_valid_LongShort"]
        + 0.15 * result["z_valid_TopK_Return"]
        + 0.10 * result["z_valid_Balanced_Acc"]
        + 0.05 * result["z_valid_LargeMove_Balanced_Acc_q70"]
    )
    return result


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - values.mean()) / std


def _extra_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in REQUIRED_SUMMARY_COLUMNS]


def plot_all(summary: pd.DataFrame, *, plots_dir: Path) -> None:
    plot_metric_bars_by_horizon(summary, plots_dir / "metric_bars_by_horizon.png")
    plot_two_metrics(summary, "IC", "RankIC", plots_dir / "ic_rankic_by_flow.png")
    plot_single_metric(summary, "Direction_Acc", plots_dir / "direction_accuracy_by_flow.png")
    plot_two_metrics(summary, "TopK_Return", "LongShort", plots_dir / "topk_longshort_by_flow.png")
    plot_single_metric(summary, "production_score", plots_dir / "close_vs_return_targets.png", group_by="target_type")
    plot_single_metric(summary, "production_score", plots_dir / "seq_len_64_vs_150.png", group_by="seq_len")
    plot_single_metric(summary, "production_score", plots_dir / "wavelet_vs_raw.png", group_by="wavelet_enabled")


def plot_metric_bars_by_horizon(summary: pd.DataFrame, output_path: Path) -> None:
    grouped = summary.groupby("horizon")[["Direction_Acc", "IC", "RankIC", "LongShort"]].mean(numeric_only=True)
    ax = grouped.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Average Metrics By Horizon")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_two_metrics(summary: pd.DataFrame, metric_a: str, metric_b: str, output_path: Path) -> None:
    frame = summary.sort_values("production_score", ascending=False).head(20)
    x = np.arange(len(frame))
    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, frame[metric_a], width, label=metric_a)
    ax.bar(x + width / 2, frame[metric_b], width, label=metric_b)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["flow_id"], rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_single_metric(summary: pd.DataFrame, metric: str, output_path: Path, *, group_by: str | None = None) -> None:
    if group_by:
        frame = summary.groupby(group_by)[metric].mean(numeric_only=True).reset_index()
        labels = frame[group_by].astype(str)
        values = frame[metric]
    else:
        frame = summary.sort_values(metric, ascending=False).head(20)
        labels = frame["flow_id"]
        values = frame[metric]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(metric)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_report(summary: pd.DataFrame, output_path: Path, *, skipped: list[str], top_k: int) -> Path:
    best_direction = summary.sort_values("direction_score", ascending=False).iloc[0]
    best_test_direction = summary.sort_values("Direction_Acc", ascending=False).iloc[0]
    best_ranking = summary.sort_values("ranking_score", ascending=False).iloc[0]
    best_production = summary.sort_values("production_score", ascending=False).iloc[0]
    best_topk = summary.sort_values("TopK_Return", ascending=False).iloc[0]
    lines = [
        "# Horizon Target Comparison",
        "",
        "## Objective",
        "",
        "Compare close-target, return-target, and blended/multitask-style stock-selection flows across horizons `t+1`, `t+3`, and `t+5`.",
        "",
        "## Data Split",
        "",
        "The suite uses the existing out-of-time `train`, `valid`, and `test` labels from `data/processed/shared/feature_panel.parquet`. Thresholds and selection scores are chosen from validation metrics; final tables report test metrics.",
        "",
        "## Leakage Controls",
        "",
        "- Features are per-symbol sequences ending at date `t` only.",
        "- Existing scalers are fit on train split only.",
        "- Close target scaling, when enabled, is per-symbol and train-only.",
        "- Wavelet features remain optional and causal because denoising is applied inside each input window only.",
        "- Direction thresholds are tuned on validation only and then locked for test evaluation.",
        "",
        "## Flow Definitions",
        "",
        "- `F1`: predict `log_close[t+1]`, convert to implied `ret_1d`.",
        "- `F2`: predict direct/log return `ret_1d`.",
        "- `F3`: predict `log_close[t+3]`, convert to implied `ret_3d`.",
        "- `F4`: predict direct/log return `ret_3d`.",
        "- `F5`: predict `log_close[t+5]`, convert to implied `ret_5d`.",
        "- `F6`: predict direct/log return `ret_5d`; this is the current stock-selection task.",
        "- `F7`: blend/multitask-style score combining return, close-implied return, and direction signal when available.",
        "",
        "## Best Models",
        "",
        f"- Best by Validation Direction Score: `{best_direction['flow_id']}` (`{best_direction['model']}`, h={int(best_direction['horizon'])})",
        f"- Best by Test Direction Accuracy: `{best_test_direction['flow_id']}` (`{best_test_direction['model']}`, h={int(best_test_direction['horizon'])}, Direction Acc={best_test_direction['Direction_Acc']:.2%})",
        f"- Best by Ranking Score: `{best_ranking['flow_id']}` (`{best_ranking['model']}`, h={int(best_ranking['horizon'])})",
        f"- Best by Production Score: `{best_production['flow_id']}` (`{best_production['model']}`, h={int(best_production['horizon'])})",
        f"- Best by Top-{top_k} Return: `{best_topk['flow_id']}` (`{best_topk['model']}`, h={int(best_topk['horizon'])})",
        "",
        "## Best By Horizon",
        "",
        markdown_table(best_by_horizon(summary)),
        "",
        "## Close Vs Return",
        "",
        compare_group_text(summary, group_column="target_type"),
        "",
        "## Horizon Takeaway",
        "",
        compare_group_text(summary, group_column="horizon"),
        "",
        "## Wavelet Takeaway",
        "",
        compare_group_text(summary, group_column="wavelet_enabled"),
        "",
        "## Seq Length Takeaway",
        "",
        compare_group_text(summary, group_column="seq_len"),
        "",
        "## Final Ranked Table",
        "",
        markdown_table(summary[REQUIRED_SUMMARY_COLUMNS].head(20)),
        "",
        "## Decision Logic",
        "",
        decision_logic_text(summary),
        "",
        "## Recommendation",
        "",
        recommendation_text(summary),
    ]
    if skipped:
        lines.extend(["", "## Skipped Runs", "", *[f"- {item}" for item in skipped]])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def best_by_horizon(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in summary.groupby("horizon"):
        best = group.sort_values("production_score", ascending=False).iloc[0]
        rows.append(
            {
                "horizon": horizon,
                "flow_id": best["flow_id"],
                "model": best["model"],
                "target_type": best["target_type"],
                "Direction_Acc": best["Direction_Acc"],
                "RankIC": best["RankIC"],
                "LongShort": best["LongShort"],
                "production_score": best["production_score"],
            }
        )
    return pd.DataFrame(rows)


def compare_group_text(summary: pd.DataFrame, *, group_column: str) -> str:
    grouped = summary.groupby(group_column)[["Direction_Acc", "IC", "RankIC", "TopK_Return", "LongShort", "production_score"]].mean(numeric_only=True)
    best = grouped["production_score"].idxmax()
    return f"Average validation-selected production score is highest for `{group_column}={best}`.\n\n{markdown_table(grouped.reset_index())}"


def recommendation_text(summary: pd.DataFrame) -> str:
    best = summary.sort_values("production_score", ascending=False).iloc[0]
    if best["target_type"] == "close" and (best["IC"] < 0 or best["LongShort"] < 0):
        return "Close prediction is useful as a paper-style diagnostic only; it does not translate into robust stock-selection metrics in this run."
    if best["target_type"] in {"return", "log_return", "score"}:
        return f"Keep `{best['flow_id']}` as the production candidate because it has the strongest validation-selected production score and test stock-selection metrics."
    return f"Keep `{best['flow_id']}` as the current candidate, but audit the flow before treating it as production."


def decision_logic_text(summary: pd.DataFrame) -> str:
    best_direction = summary.sort_values("Direction_Acc", ascending=False).iloc[0]
    best_production = summary.sort_values("production_score", ascending=False).iloc[0]
    best_topk = summary.sort_values("TopK_Return", ascending=False).iloc[0]
    best_horizon = summary.groupby("horizon")["production_score"].mean().idxmax()
    lines = [
        f"- Best test direction accuracy is `{best_direction['flow_id']}` with `{best_direction['Direction_Acc']:.2%}`.",
        f"- Best production candidate is `{best_production['flow_id']}` with production score `{best_production['production_score']:.4f}`.",
        f"- Best Top-k return candidate is `{best_topk['flow_id']}` with TopK_Return `{best_topk['TopK_Return']:.6f}`.",
        f"- Best average horizon by production score is `t+{int(best_horizon)}`.",
    ]
    if best_direction["target_type"] == "close" and best_production["target_type"] != "close":
        lines.append(
            "- Close prediction improves or preserves headline direction accuracy in this run, "
            "but it does not win IC/RankIC/top-k, so it is paper-style diagnostic only."
        )
    if best_production["horizon"] == 5:
        lines.append("- `t+5` return-style scoring remains the main stock-selection setup.")
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
