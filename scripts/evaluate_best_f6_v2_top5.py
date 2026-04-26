from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


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
from vnstock.models.common.sequence_data import build_effective_feature_columns, build_scaled_sequence_splits, load_shared_feature_panel  # noqa: E402
from vnstock.models.common.torch_training import predict_multitask_model, resolve_device  # noqa: E402
from vnstock.models.itransformer.model import build_model as build_itransformer_model  # noqa: E402
from vnstock.models.kronos.zero_shot import run_kronos_zero_shot  # noqa: E402
from vnstock.models.xlstm_ts.model import build_model as build_xlstm_model  # noqa: E402
from vnstock.pipelines.run_investment_backtest import (  # noqa: E402
    _markdown_table,
    _prepare_frame,
    compute_cross_section_stats,
    compute_score_bucket_returns,
    run_backtest,
)
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402
from vnstock.utils.schema import PredictionContext  # noqa: E402


TOP_K = 5
INITIAL_CAPITAL = 100_000_000.0
TRANSACTION_COST_BPS = 15.0
REBALANCE_EVERY = 5
BEST_PREDICTIONS = ROOT / "outputs" / "final" / "best_f6_v2_predictions.parquet"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "best_f6_v2_top5"
FIGURE_DIR = ROOT / "outputs" / "figures" / "best_f6_v2_top5"
MODEL_COMPARE_DIR = ROOT / "outputs" / "final" / "model_compare_top5"


def main() -> None:
    clean_outputs()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)
    ensure_dir(MODEL_COMPARE_DIR)

    best_frame = _rename_best_model(load_table(BEST_PREDICTIONS))
    save_table(best_frame, BEST_PREDICTIONS)

    invest_summary, trades, equity, buckets = run_final_backtest(best_frame)
    model_predictions = build_model_comparison_predictions(best_frame)
    model_compare = build_model_comparison(model_predictions)

    write_final_report(invest_summary, buckets, model_compare)
    plot_equity_curve(equity)
    plot_score_buckets(buckets)
    plot_model_compare(model_compare)

    print("Final top-5 evaluation complete.")
    print(OUTPUT_DIR / "best_f6_v2_top5_summary.csv")
    print(OUTPUT_DIR / "best_f6_v2_top5_model_compare.csv")
    print(OUTPUT_DIR / "best_f6_v2_top5_report.md")


def clean_outputs() -> None:
    for target in [
        OUTPUT_DIR,
        FIGURE_DIR,
        MODEL_COMPARE_DIR,
    ]:
        if target.exists():
            _safe_rmtree(target)
    predictions_root = ROOT / "outputs" / "predictions"
    if predictions_root.exists():
        _safe_rmtree(predictions_root)


def run_final_backtest(best_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_frame = _prepare_frame(best_frame, split="test", score_column="y_pred")
    cross_section = compute_cross_section_stats(test_frame, score_column="y_pred", top_k=TOP_K)
    buckets = compute_score_bucket_returns(test_frame, score_column="y_pred", buckets=5)

    summaries: list[dict[str, object]] = []
    trades: list[pd.DataFrame] = []
    equities: list[pd.DataFrame] = []
    for mode in ("long-only", "long-short"):
        result = run_backtest(
            test_frame,
            mode=mode,
            score_column="y_pred",
            top_k=TOP_K,
            rebalance_every=REBALANCE_EVERY,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            cross_section=cross_section,
        )
        summaries.append(result.summary)
        trades.append(result.trades)
        equities.append(result.equity_curve)

    summary_frame = pd.DataFrame(summaries).sort_values("mode").reset_index(drop=True)
    trades_frame = pd.concat(trades, ignore_index=True)
    equity_frame = pd.concat(equities, ignore_index=True)
    save_table(summary_frame, OUTPUT_DIR / "best_f6_v2_top5_summary.csv")
    save_table(trades_frame, OUTPUT_DIR / "best_f6_v2_top5_trades.csv")
    save_table(equity_frame, OUTPUT_DIR / "best_f6_v2_top5_equity_curve.csv")
    save_table(buckets, OUTPUT_DIR / "best_f6_v2_top5_score_buckets.csv")
    return summary_frame, trades_frame, equity_frame, buckets


def build_model_comparison_predictions(best_frame: pd.DataFrame) -> list[pd.DataFrame]:
    predictions = [
        _best_test_predictions(best_frame),
        infer_checkpoint_model(
            model_name="xlstm_ts",
            model_label="xLSTM-TS pure",
            checkpoint_dir=ROOT / "registry" / "models" / "xlstm_ts" / "xlstm_ts_20260424T054331Z",
        ),
        infer_checkpoint_model(
            model_name="itransformer",
            model_label="iTransformer pure",
            checkpoint_dir=ROOT / "registry" / "models" / "itransformer" / "itransformer_20260424T060006Z",
        ),
    ]
    kronos = infer_kronos_latest()
    if kronos is not None:
        predictions.append(kronos)
    for frame in predictions:
        save_table(frame, MODEL_COMPARE_DIR / f"{_slug(str(frame['model_family'].iloc[0]))}_predictions.parquet")
    return predictions


def build_model_comparison(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    rows = [compute_prediction_metrics(frame, top_k=TOP_K) for frame in predictions]
    output = pd.DataFrame(rows).sort_values(["coverage_scope", "LongShort5"], ascending=[True, False])
    save_table(output, OUTPUT_DIR / "best_f6_v2_top5_model_compare.csv")
    return output


def infer_checkpoint_model(*, model_name: str, model_label: str, checkpoint_dir: Path) -> pd.DataFrame:
    manifest = _read_json(checkpoint_dir / "manifest.json")
    checkpoint = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    config = dict(checkpoint.get("config") or manifest["config"])
    feature_columns = list(checkpoint.get("feature_columns") or config["feature_columns"])
    wavelet_config = config.get("wavelet_denoise")
    target_scaling = config.get("target_scaling")
    target_column = str(config["target"])
    lookback = int(config.get("context_length", config.get("seq_len")))

    panel = load_shared_feature_panel()
    sequence_splits, _ = build_scaled_sequence_splits(
        feature_panel=panel,
        feature_columns=feature_columns,
        target_column=target_column,
        lookback=lookback,
        wavelet_config=wavelet_config,
        target_scaling=target_scaling,
    )
    effective_features = build_effective_feature_columns(feature_columns, wavelet_config)
    if model_name == "xlstm_ts":
        model = build_xlstm_model(config, num_features=len(effective_features))
    elif model_name == "itransformer":
        model = build_itransformer_model(config)
    else:
        raise ValueError(f"Unsupported checkpoint model: {model_name}")

    state_dict = remap_legacy_single_head(checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"{model_label}: missing checkpoint keys ignored: {len(missing)}")
    if unexpected:
        print(f"{model_label}: unexpected checkpoint keys ignored: {len(unexpected)}")

    device = resolve_device("cpu")
    model.to(device)
    sample_set = sequence_splits["test"]
    result = predict_multitask_model(model, sample_set, batch_size=1024, device=device, input_dtype=torch.float32)
    frame = sample_set.meta.copy()
    target_values, prediction_values = restore_target_scale(sample_set, result.y_pred)
    frame[target_column] = target_values
    frame["y_pred"] = prediction_values
    context = PredictionContext(
        model_family=model_label,
        model_version=str(manifest.get("run_id", checkpoint_dir.name)),
        target_name=target_column,
        horizon=int(config.get("horizon", 5)),
        run_id=str(manifest.get("run_id", checkpoint_dir.name)),
    )
    output = standardize_prediction_output(frame, context, y_true_column=target_column)
    output["direction_score_column"] = "y_pred"
    output["direction_threshold"] = 0.0
    output["coverage_scope"] = "full_test"
    return output


def infer_kronos_latest() -> pd.DataFrame | None:
    try:
        output_path = run_kronos_zero_shot(
            {
                "pretrained_model": "NeoQuasar/Kronos-small",
                "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                "context_length": 64,
                "prediction_length": 5,
                "split": "test",
                "latest_only": True,
                "sample_count": 1,
                "temperature": 1.0,
                "top_p": 0.9,
                "device": "cpu",
                "kronos_repo": str(ROOT / "external" / "Kronos"),
            }
        )
    except Exception as exc:
        print(f"Kronos latest-only inference skipped: {exc}")
        return None
    frame = load_table(output_path)
    frame["model_family"] = "Kronos zero-shot"
    frame["coverage_scope"] = "latest_only"
    frame["direction_score_column"] = "y_pred"
    frame["direction_threshold"] = 0.0
    return frame


def compute_prediction_metrics(frame: pd.DataFrame, *, top_k: int) -> dict[str, object]:
    test = frame.loc[frame["split"].astype(str) == "test"].dropna(subset=["date", "symbol", "y_true", "y_pred"]).copy()
    test["date"] = pd.to_datetime(test["date"])
    cross_section = compute_cross_section_stats(test, score_column="y_pred", top_k=top_k)
    y_true = test["y_true"].to_numpy(dtype=float)
    y_score = test["y_pred"].to_numpy(dtype=float)
    threshold = _direction_threshold(test)
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


def write_final_report(invest_summary: pd.DataFrame, buckets: pd.DataFrame, model_compare: pd.DataFrame) -> None:
    invest_cols = [
        "mode",
        "final_capital",
        "total_profit",
        "total_return",
        "benchmark_total_return",
        "excess_profit_vs_benchmark",
        "sharpe_proxy",
        "hit_rate",
        "max_drawdown",
    ]
    compare_cols = [
        "model",
        "coverage_scope",
        "rows",
        "dates",
        "symbols",
        "IC",
        "RankIC",
        "Direction_Acc",
        "Balanced_Acc",
        "Top5_Return",
        "LongShort5",
        "Top5_Direction_Acc",
    ]
    lines = [
        "# BestF6-v2 Top-5 Final Evaluation",
        "",
        "Final setting: `top_k=5`, `rebalance_every=5`, `transaction_cost=15 bps`.",
        "",
        "## Investment P/L Proxy",
        "",
        _markdown_table(invest_summary[invest_cols]),
        "",
        "## Score Buckets",
        "",
        _markdown_table(buckets),
        "",
        "## Model Comparison",
        "",
        _markdown_table(model_compare[compare_cols]),
        "",
        "Kronos is latest-only because the current zero-shot adapter forecasts the latest test window per symbol.",
        "It is useful as a point-in-time reference, not a full historical benchmark.",
        "",
        "## Artifacts",
        "",
        "- `outputs/final/best_f6_v2_predictions.parquet`",
        "- `outputs/final/model_compare_top5/`",
        "- `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_summary.csv`",
        "- `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_model_compare.csv`",
        "- `outputs/figures/best_f6_v2_top5/`",
    ]
    (OUTPUT_DIR / "best_f6_v2_top5_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_equity_curve(equity: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 5))
    for mode, group in equity.groupby("mode"):
        ordered = group.sort_values("date")
        plt.plot(ordered["date"], ordered["equity"], label=f"{mode} equity")
    benchmark = equity.loc[equity["mode"] == equity["mode"].iloc[0]].sort_values("date")
    plt.plot(benchmark["date"], benchmark["benchmark_equity"], label="equal-weight benchmark", color="black", linestyle="--")
    plt.title("BestF6-v2 top-5 equity curve")
    plt.xlabel("date")
    plt.ylabel("capital")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_f6_v2_top5_equity_curve.png", dpi=160)
    plt.close()


def plot_score_buckets(buckets: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(buckets["score_bucket"].astype(str), buckets["avg_realized_return"])
    plt.title("BestF6-v2 score buckets: realized t+5 return")
    plt.xlabel("score bucket, low to high")
    plt.ylabel("avg realized return")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_f6_v2_top5_score_buckets.png", dpi=160)
    plt.close()


def plot_model_compare(model_compare: pd.DataFrame) -> None:
    full = model_compare.copy()
    plt.figure(figsize=(10, 5))
    plt.bar(full["model"], full["LongShort5"])
    plt.title("Top-5 long-short spread by model")
    plt.ylabel("mean t+5 spread")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "best_f6_v2_model_longshort5.png", dpi=160)
    plt.close()


def restore_target_scale(sample_set: Any, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sample_set.meta.empty or not {"target_raw", "target_mean", "target_std"}.issubset(sample_set.meta.columns):
        return sample_set.y, predictions
    means = sample_set.meta["target_mean"].to_numpy(dtype=float)
    stds = sample_set.meta["target_std"].to_numpy(dtype=float)
    target_values = sample_set.meta["target_raw"].to_numpy(dtype=float)
    prediction_values = predictions * stds + means
    return target_values, prediction_values


def _direction_threshold(frame: pd.DataFrame) -> float:
    if "direction_threshold" in frame.columns and frame["direction_threshold"].notna().any():
        return float(frame["direction_threshold"].dropna().iloc[0])
    return 0.0


def remap_legacy_single_head(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Old registry checkpoints used `head.*`; current multitask models use `return_head.*`."""
    output = dict(state_dict)
    for key, value in list(state_dict.items()):
        if key.startswith("head."):
            output[f"return_head.{key.removeprefix('head.')}"] = value
    return output


def _best_test_predictions(best_frame: pd.DataFrame) -> pd.DataFrame:
    output = best_frame.copy()
    output["model_family"] = "BestF6-v2"
    output["model_version"] = "direction_excess_blend_top5"
    output["coverage_scope"] = "full_test"
    output["direction_score_column"] = "y_pred"
    return output


def _rename_best_model(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["model_family"] = "BestF6-v2"
    output["model_version"] = "direction_excess_blend_top5"
    output["run_id"] = "best_f6_v2_top5_final"
    return output


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_rmtree(path: Path) -> None:
    full = path.resolve()
    root = ROOT.resolve()
    if root not in full.parents and full != root:
        raise ValueError(f"Refusing to delete outside repo: {full}")
    if path.is_dir():
        import shutil

        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


if __name__ == "__main__":
    main()
