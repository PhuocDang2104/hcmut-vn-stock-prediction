from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
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
from vnstock.pipelines.run_investment_backtest import _markdown_table, compute_cross_section_stats  # noqa: E402
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402


BASELINE_SOURCE = ROOT / "outputs" / "final" / "model_suite_top5" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"
BASELINE_ALIAS = ROOT / "outputs" / "final" / "hybrid_xlstm_baseline_predictions.parquet"
PRODUCTION_ALIAS = ROOT / "outputs" / "final" / "hybrid_xlstm_long_only_production_predictions.parquet"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "long_only_portfolio_layer"
FIGURE_DIR = ROOT / "outputs" / "figures" / "long_only_portfolio_layer"

TOP_K = 5
POOL_SIZE = 10
REBALANCE_EVERY = 5
INITIAL_CAPITAL = 100_000_000.0
TRANSACTION_COST_BPS = 15.0


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    weights: dict[str, float]


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    filter_mode: str = "none"
    sizing_mode: str = "equal"
    exposure_mode: str = "always_100"


def main() -> None:
    clean_outputs()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(FIGURE_DIR)

    frame = prepare_frame()
    thresholds = build_thresholds(frame)
    score_specs = build_score_specs()
    portfolio_specs = build_portfolio_specs()

    valid_rows = evaluate_grid(frame, score_specs, portfolio_specs, split="valid", thresholds=thresholds)
    valid_metrics = add_long_only_selection_score(pd.DataFrame(valid_rows))
    winner = valid_metrics.sort_values("selection_score", ascending=False).iloc[0]
    selected_score = next(spec for spec in score_specs if spec.name == winner["score_spec"])
    selected_portfolio = next(spec for spec in portfolio_specs if spec.name == winner["portfolio_spec"])

    test_rows = evaluate_selected_and_baseline(
        frame,
        selected_score=selected_score,
        selected_portfolio=selected_portfolio,
        thresholds=thresholds,
    )
    test_metrics = pd.DataFrame(test_rows)

    baseline_predictions = build_prediction_output(frame, build_score(frame, build_score_specs()[0]), "Hybrid xLSTM Direction-Excess Blend")
    selected_predictions = build_prediction_output(
        frame,
        build_score(frame, selected_score),
        "Hybrid xLSTM Long-Only Portfolio Layer",
    )
    production_name, production_predictions = choose_production_predictions(test_metrics, baseline_predictions, selected_predictions)
    save_table(baseline_predictions, BASELINE_ALIAS)
    save_table(production_predictions, PRODUCTION_ALIAS)

    save_table(valid_metrics.sort_values("selection_score", ascending=False), OUTPUT_DIR / "valid_long_only_grid.csv")
    save_table(test_metrics, OUTPUT_DIR / "test_selected_vs_baseline.csv")
    write_selected_config(winner, selected_score, selected_portfolio, thresholds, production_name)
    write_report(valid_metrics, test_metrics, selected_score, selected_portfolio, production_name)
    plot_test_metrics(test_metrics)

    print(test_metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def prepare_frame() -> pd.DataFrame:
    predictions = load_table(BASELINE_SOURCE).copy()
    panel = load_table(ROOT / "data" / "processed" / "shared" / "feature_panel.parquet").copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = add_downside_features(panel)
    columns = [
        "symbol",
        "date",
        "ret_20d",
        "excess_ret_5d",
        "relative_strength_20d",
        "ma_ratio_5_20",
        "rolling_vol_20",
        "downside_vol_20",
        "volume_zscore_20",
        "gap_rel",
        "market_ret_20d",
        "market_vol_20",
    ]
    frame = predictions.merge(panel[columns], on=["symbol", "date"], how="left")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["base_score"] = frame["y_pred"].astype(float)
    for component in (
        "base_score",
        "relative_strength_20d",
        "excess_ret_5d",
        "downside_vol_20",
        "rolling_vol_20",
        "ret_20d",
    ):
        frame[f"z_{component}"] = frame.groupby("date")[component].transform(cross_section_zscore)
    return frame


def add_downside_features(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values(["symbol", "date"]).copy()
    output["downside_ret_1d"] = output["ret_1d"].clip(upper=0.0)
    output["downside_vol_20"] = (
        output.groupby("symbol")["downside_ret_1d"]
        .rolling(window=20, min_periods=5)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    output["downside_vol_20"] = output.groupby("symbol")["downside_vol_20"].ffill().fillna(0.0)
    return output


def build_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    train_valid = frame.loc[frame["split"].astype(str).isin(["train", "valid"])].copy()
    by_date = train_valid.groupby("date", as_index=False).agg(
        market_ret_20d=("market_ret_20d", "mean"),
        market_vol_20=("market_vol_20", "mean"),
    )
    confidence_values = []
    valid = frame.loc[frame["split"].astype(str).eq("valid")].copy()
    for _, group in valid.groupby("date"):
        ranked = group.sort_values("base_score", ascending=False)
        if len(ranked) >= 15:
            confidence_values.append(float(ranked.head(5)["base_score"].mean() - ranked.iloc[5:15]["base_score"].mean()))
    return {
        "market_vol_median": float(by_date["market_vol_20"].median()),
        "confidence_median": float(np.median(confidence_values)) if confidence_values else 0.0,
    }


def build_score_specs() -> list[ScoreSpec]:
    return [
        ScoreSpec("baseline_only", {"base_score": 1.0}),
        ScoreSpec("baseline_rs20_005", {"base_score": 0.95, "relative_strength_20d": 0.05}),
        ScoreSpec("baseline_rs20_010", {"base_score": 0.90, "relative_strength_20d": 0.10}),
        ScoreSpec("baseline_excess5_005", {"base_score": 0.95, "excess_ret_5d": 0.05}),
        ScoreSpec("baseline_excess5_010", {"base_score": 0.90, "excess_ret_5d": 0.10}),
        ScoreSpec("baseline_downside_005", {"base_score": 1.00, "downside_vol_20": -0.05}),
        ScoreSpec("baseline_downside_010", {"base_score": 1.00, "downside_vol_20": -0.10}),
        ScoreSpec(
            "baseline_rs20_downside",
            {"base_score": 0.95, "relative_strength_20d": 0.05, "downside_vol_20": -0.05},
        ),
        ScoreSpec(
            "baseline_excess5_downside",
            {"base_score": 0.95, "excess_ret_5d": 0.05, "downside_vol_20": -0.05},
        ),
    ]


def build_portfolio_specs() -> list[PortfolioSpec]:
    return [
        PortfolioSpec("equal_top5"),
        PortfolioSpec("rank_weighted", sizing_mode="rank_weighted"),
        PortfolioSpec("risk_adjusted", sizing_mode="risk_adjusted"),
        PortfolioSpec("filter_downside", filter_mode="downside"),
        PortfolioSpec("filter_momentum", filter_mode="momentum"),
        PortfolioSpec("filter_liquidity", filter_mode="liquidity"),
        PortfolioSpec("filter_combo", filter_mode="combo"),
        PortfolioSpec("cash_regime_75", exposure_mode="weak_regime_75"),
        PortfolioSpec("cash_regime_50", exposure_mode="weak_regime_50"),
        PortfolioSpec("confidence_cash_50", exposure_mode="confidence_50"),
    ]


def evaluate_grid(
    frame: pd.DataFrame,
    score_specs: list[ScoreSpec],
    portfolio_specs: list[PortfolioSpec],
    *,
    split: str,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score_spec in score_specs:
        scored = frame.copy()
        scored["candidate_score"] = build_score(scored, score_spec)
        prediction_metrics = compute_prediction_metrics(scored, split=split, score_column="candidate_score")
        for portfolio_spec in portfolio_specs:
            backtest = run_long_only_backtest(
                scored,
                split=split,
                score_column="candidate_score",
                portfolio_spec=portfolio_spec,
                thresholds=thresholds,
            )
            rows.append(
                {
                    "score_spec": score_spec.name,
                    "portfolio_spec": portfolio_spec.name,
                    **prediction_metrics,
                    **backtest,
                }
            )
    return rows


def evaluate_selected_and_baseline(
    frame: pd.DataFrame,
    *,
    selected_score: ScoreSpec,
    selected_portfolio: PortfolioSpec,
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    baseline_score = build_score_specs()[0]
    baseline_portfolio = build_portfolio_specs()[0]
    rows: list[dict[str, object]] = []
    for label, score_spec, portfolio_spec in (
        ("baseline_current", baseline_score, baseline_portfolio),
        ("selected_long_only", selected_score, selected_portfolio),
    ):
        scored = frame.copy()
        scored["candidate_score"] = build_score(scored, score_spec)
        prediction_metrics = compute_prediction_metrics(scored, split="test", score_column="candidate_score")
        backtest = run_long_only_backtest(
            scored,
            split="test",
            score_column="candidate_score",
            portfolio_spec=portfolio_spec,
            thresholds=thresholds,
        )
        rows.append(
            {
                "candidate": label,
                "score_spec": score_spec.name,
                "portfolio_spec": portfolio_spec.name,
                **prediction_metrics,
                **backtest,
            }
        )
    return rows


def build_score(frame: pd.DataFrame, spec: ScoreSpec) -> np.ndarray:
    score = np.zeros(len(frame), dtype=float)
    for component, weight in spec.weights.items():
        score += float(weight) * frame[f"z_{component}"].fillna(0.0).to_numpy(dtype=float)
    return score.astype(np.float32, copy=False)


def run_long_only_backtest(
    frame: pd.DataFrame,
    *,
    split: str,
    score_column: str,
    portfolio_spec: PortfolioSpec,
    thresholds: dict[str, float],
) -> dict[str, float]:
    data = frame.loc[frame["split"].astype(str).eq(split)].sort_values("date").copy()
    rebalance_dates = sorted(data["date"].unique())[::REBALANCE_EVERY]
    cost_rate = TRANSACTION_COST_BPS / 10_000.0
    equity = INITIAL_CAPITAL
    benchmark_equity = INITIAL_CAPITAL
    peak = equity
    previous_weights: dict[str, float] = {}
    returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    exposures: list[float] = []
    drawdowns: list[float] = []
    hit_rows: list[float] = []

    for date in rebalance_dates:
        group = data.loc[data["date"].eq(date)].dropna(subset=["y_true", score_column]).copy()
        if group.empty:
            continue
        selected = select_long_names(group, score_column=score_column, portfolio_spec=portfolio_spec)
        exposure = compute_exposure(group, selected, score_column=score_column, portfolio_spec=portfolio_spec, thresholds=thresholds)
        weights = compute_weights(selected, portfolio_spec=portfolio_spec, exposure=exposure)
        gross_return = float(sum(weights.get(str(row.symbol), 0.0) * float(row.y_true) for row in selected.itertuples()))
        turnover = portfolio_turnover(previous_weights, weights)
        net_return = gross_return - turnover * cost_rate
        benchmark_return = float(group["y_true"].mean())
        equity *= 1.0 + net_return
        benchmark_equity *= 1.0 + benchmark_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        returns.append(net_return)
        gross_returns.append(gross_return)
        turnovers.append(turnover)
        exposures.append(exposure)
        drawdowns.append(drawdown)
        hit_rows.append(float((selected["y_true"] > 0).mean()))
        previous_weights = weights

    values = np.asarray(returns, dtype=float)
    gross = np.asarray(gross_returns, dtype=float)
    turnover_values = np.asarray(turnovers, dtype=float)
    exposure_values = np.asarray(exposures, dtype=float)
    periods_per_year = 252.0 / REBALANCE_EVERY
    return {
        "periods": float(len(values)),
        "final_capital": equity,
        "total_return": equity / INITIAL_CAPITAL - 1.0,
        "total_profit": equity - INITIAL_CAPITAL,
        "benchmark_total_return": benchmark_equity / INITIAL_CAPITAL - 1.0,
        "excess_return_vs_benchmark": equity / INITIAL_CAPITAL - benchmark_equity / INITIAL_CAPITAL,
        "avg_gross_period_return": float(np.mean(gross)) if len(gross) else math.nan,
        "avg_net_period_return": float(np.mean(values)) if len(values) else math.nan,
        "period_volatility": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
        "sharpe_proxy": sharpe(values, periods_per_year),
        "hit_rate": float(np.mean(values > 0)) if len(values) else math.nan,
        "avg_top5_hit_rate": float(np.mean(hit_rows)) if hit_rows else math.nan,
        "max_drawdown": float(np.min(drawdowns)) if drawdowns else math.nan,
        "avg_turnover": float(np.mean(turnover_values)) if len(turnover_values) else math.nan,
        "avg_exposure": float(np.mean(exposure_values)) if len(exposure_values) else math.nan,
    }


def select_long_names(group: pd.DataFrame, *, score_column: str, portfolio_spec: PortfolioSpec) -> pd.DataFrame:
    ranked = group.sort_values(score_column, ascending=False).copy()
    pool = ranked.head(min(POOL_SIZE, len(ranked))).copy()
    filtered = apply_filter(pool, group, mode=portfolio_spec.filter_mode)
    if len(filtered) < TOP_K:
        missing = ranked.loc[~ranked["symbol"].isin(filtered["symbol"])].head(TOP_K - len(filtered))
        filtered = pd.concat([filtered, missing], ignore_index=True)
    return filtered.sort_values(score_column, ascending=False).head(TOP_K).copy()


def apply_filter(pool: pd.DataFrame, universe: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    if mode == "none":
        return pool
    keep = pd.Series(True, index=pool.index)
    if mode in {"downside", "combo"}:
        downside_threshold = universe["downside_vol_20"].quantile(0.80)
        keep &= pool["downside_vol_20"].fillna(0.0) <= downside_threshold
    if mode in {"momentum", "combo"}:
        rs_threshold = universe["relative_strength_20d"].quantile(0.40)
        ma_threshold = universe["ma_ratio_5_20"].quantile(0.40)
        keep &= pool["relative_strength_20d"].fillna(-np.inf) >= rs_threshold
        keep &= pool["ma_ratio_5_20"].fillna(-np.inf) >= ma_threshold
    if mode in {"liquidity", "combo"}:
        keep &= pool["volume_zscore_20"].fillna(-np.inf) >= -1.0
    return pool.loc[keep].copy()


def compute_exposure(
    group: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    score_column: str,
    portfolio_spec: PortfolioSpec,
    thresholds: dict[str, float],
) -> float:
    if portfolio_spec.exposure_mode == "always_100":
        return 1.0
    if portfolio_spec.exposure_mode in {"weak_regime_75", "weak_regime_50"}:
        weak = float(group["market_ret_20d"].mean()) < 0.0 and float(group["market_vol_20"].mean()) > thresholds["market_vol_median"]
        if not weak:
            return 1.0
        return 0.75 if portfolio_spec.exposure_mode == "weak_regime_75" else 0.50
    if portfolio_spec.exposure_mode == "confidence_50":
        ranked = group.sort_values(score_column, ascending=False)
        if len(ranked) < 15:
            return 1.0
        confidence = float(ranked.head(5)[score_column].mean() - ranked.iloc[5:15][score_column].mean())
        return 0.50 if confidence < thresholds["confidence_median"] else 1.0
    return float(selected.attrs.get("exposure", 1.0))


def compute_weights(selected: pd.DataFrame, *, portfolio_spec: PortfolioSpec, exposure: float) -> dict[str, float]:
    symbols = selected["symbol"].astype(str).tolist()
    if not symbols:
        return {}
    if portfolio_spec.sizing_mode == "rank_weighted":
        base = np.asarray([0.25, 0.22, 0.20, 0.18, 0.15][: len(symbols)], dtype=float)
    elif portfolio_spec.sizing_mode == "risk_adjusted":
        risk = selected["downside_vol_20"].fillna(selected["downside_vol_20"].median()).to_numpy(dtype=float)
        base = 1.0 / np.clip(risk, 1e-4, None)
        base = np.clip(base / base.sum(), 0.10, 0.30)
    else:
        base = np.ones(len(symbols), dtype=float)
    base = base / base.sum() * exposure
    return dict(zip(symbols, base, strict=False))


def portfolio_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous) | set(current)
    return float(sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


def compute_prediction_metrics(frame: pd.DataFrame, *, split: str, score_column: str) -> dict[str, float]:
    data = frame.loc[frame["split"].astype(str).eq(split)].dropna(subset=["date", "symbol", "y_true", score_column]).copy()
    cross_section = compute_cross_section_stats(data, score_column=score_column, top_k=TOP_K)
    y_true = data["y_true"].to_numpy(dtype=float)
    y_score = data[score_column].to_numpy(dtype=float)
    threshold = tune_threshold(y_true, y_score)
    return {
        "rows": float(len(data)),
        "dates": float(data["date"].nunique()),
        "symbols": float(data["symbol"].nunique()),
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
        "Top5_Direction_Acc": cross_section["TopK_Direction_Acc"],
        "LongShort5_Diagnostic": cross_section["LongShort"],
    }


def add_long_only_selection_score(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    weights = {
        "total_return": 0.30,
        "sharpe_proxy": 0.20,
        "max_drawdown_abs": -0.15,
        "avg_turnover": -0.10,
        "Top5_Direction_Acc": 0.15,
        "RankIC": 0.10,
    }
    output["max_drawdown_abs"] = output["max_drawdown"].abs()
    score = pd.Series(0.0, index=output.index)
    for column, weight in weights.items():
        values = output[column].astype(float)
        std = values.std(ddof=0)
        normalized = (values - values.mean()) / std if std and np.isfinite(std) else values * 0.0
        score += weight * normalized
    output["selection_score"] = score
    return output


def build_prediction_output(frame: pd.DataFrame, score: np.ndarray, model_family: str) -> pd.DataFrame:
    output = frame.copy()
    output["model_family"] = model_family
    output["model_version"] = "long_only_core_score_v1"
    output["run_id"] = "long_only_core_score_v1"
    output["y_pred"] = score
    output["direction_score"] = score
    output["direction_threshold"] = 0.0
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


def choose_production_predictions(
    test_metrics: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    selected_predictions: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    baseline = test_metrics.loc[test_metrics["candidate"].eq("baseline_current")].iloc[0]
    selected = test_metrics.loc[test_metrics["candidate"].eq("selected_long_only")].iloc[0]
    selected_wins_return = float(selected["total_return"]) > float(baseline["total_return"])
    selected_keeps_hit_rate = float(selected["Top5_Direction_Acc"]) >= float(baseline["Top5_Direction_Acc"]) - 0.002
    if selected_wins_return and selected_keeps_hit_rate:
        return "selected_long_only", selected_predictions
    return "baseline_current", baseline_predictions


def write_selected_config(
    winner: pd.Series,
    selected_score: ScoreSpec,
    selected_portfolio: PortfolioSpec,
    thresholds: dict[str, float],
    production_name: str,
) -> None:
    payload: dict[str, Any] = {
        "selected_score_spec": selected_score.name,
        "selected_score_weights": selected_score.weights,
        "selected_portfolio_spec": selected_portfolio.__dict__,
        "selection_split": "valid",
        "selection_objective": {
            "total_return": 0.30,
            "sharpe_proxy": 0.20,
            "max_drawdown_abs": -0.15,
            "avg_turnover": -0.10,
            "Top5_Direction_Acc": 0.15,
            "RankIC": 0.10,
        },
        "thresholds": thresholds,
        "production_decision": production_name,
        "valid_winner": winner.to_dict(),
    }
    (OUTPUT_DIR / "selected_long_only_config.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_report(
    valid_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    selected_score: ScoreSpec,
    selected_portfolio: PortfolioSpec,
    production_name: str,
) -> None:
    top_valid = valid_metrics.sort_values("selection_score", ascending=False).head(12)
    metric_cols = [
        "score_spec",
        "portfolio_spec",
        "IC",
        "RankIC",
        "Top5_Return",
        "Top5_Direction_Acc",
        "total_return",
        "sharpe_proxy",
        "max_drawdown",
        "avg_turnover",
        "avg_exposure",
        "selection_score",
    ]
    test_cols = [
        "candidate",
        "score_spec",
        "portfolio_spec",
        "IC",
        "RankIC",
        "Top5_Return",
        "Top5_Direction_Acc",
        "total_return",
        "final_capital",
        "sharpe_proxy",
        "max_drawdown",
        "avg_turnover",
        "avg_exposure",
    ]
    lines = [
        "# Hybrid xLSTM Long-Only Portfolio Layer",
        "",
        "Goal: keep the Hybrid xLSTM baseline as the core ranking signal and test only long-only portfolio-layer improvements.",
        "",
        "Selection is done on validation using long-only after-cost metrics. `LongShort5` is diagnostic only and is not used as the main selection objective.",
        "",
        "Selected configuration:",
        "",
        f"- Score spec: `{selected_score.name}` with weights `{json.dumps(selected_score.weights, sort_keys=True)}`",
        f"- Portfolio spec: `{selected_portfolio.name}`",
        "",
        "Selection score:",
        "",
        "```text",
        "0.30 * z(total_return) + 0.20 * z(sharpe_proxy)",
        "  - 0.15 * z(abs(max_drawdown)) - 0.10 * z(avg_turnover)",
        "  + 0.15 * z(Top5_Direction_Acc) + 0.10 * z(RankIC)",
        "```",
        "",
        "## Top Validation Candidates",
        "",
        _markdown_table(top_valid[metric_cols]),
        "",
        "## Locked Test Comparison",
        "",
        _markdown_table(test_metrics[test_cols]),
        "",
        "## Decision",
        "",
        f"Production decision: `{production_name}`.",
        "",
        "Promotion rule: the selected candidate must improve locked-test long-only total return and keep Top5 Direction Accuracy approximately unchanged. Otherwise baseline remains production and the selected candidate is only a defensive overlay idea.",
        "",
        "Artifacts:",
        "",
        "- `outputs/final/hybrid_xlstm_baseline_predictions.parquet`",
        "- `outputs/final/hybrid_xlstm_long_only_production_predictions.parquet`",
        "- `outputs/reports/long_only_portfolio_layer/valid_long_only_grid.csv`",
        "- `outputs/reports/long_only_portfolio_layer/test_selected_vs_baseline.csv`",
        "- `outputs/reports/long_only_portfolio_layer/selected_long_only_config.json`",
        "- `outputs/figures/long_only_portfolio_layer/test_total_return.png`",
    ]
    (OUTPUT_DIR / "long_only_portfolio_layer_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_test_metrics(test_metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(test_metrics["candidate"], test_metrics["total_return"])
    plt.title("Long-only test total return after cost")
    plt.ylabel("total return")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "test_total_return.png", dpi=160)
    plt.close()


def cross_section_zscore(values: pd.Series) -> pd.Series:
    values = values.fillna(0.0).astype(float)
    std = values.std(ddof=0)
    if not std or not np.isfinite(std):
        return values * 0.0
    return (values - values.mean()) / std


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


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return math.nan
    std = float(np.std(returns, ddof=1))
    if std <= 1e-12:
        return math.nan
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


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
