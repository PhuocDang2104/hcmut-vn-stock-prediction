from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vnstock.utils.io import ensure_dir, load_table, save_table
from vnstock.utils.logging import get_logger


DEFAULT_PREDICTIONS = "outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate portfolio P/L from BestF6-v2 stock-selection predictions. "
            "This is a return-based proxy, not an executable brokerage backtest."
        )
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modes", nargs="+", choices=["long-only", "long-short"], default=["long-only", "long-short"])
    parser.add_argument("--score-column", default="y_pred")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=100_000_000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=15.0)
    parser.add_argument("--output-dir", default="outputs/reports/best_f6_v2_top5")
    parser.add_argument("--prefix", default="best_f6_v2_top5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_investment_backtest")
    frame = _prepare_frame(load_table(args.predictions), split=args.split, score_column=args.score_column)
    output_dir = ensure_dir(args.output_dir)

    cross_section = compute_cross_section_stats(frame, score_column=args.score_column, top_k=args.top_k)
    deciles = compute_score_bucket_returns(frame, score_column=args.score_column, buckets=5)

    summaries: list[dict[str, object]] = []
    trades: list[pd.DataFrame] = []
    equity_curves: list[pd.DataFrame] = []
    for mode in args.modes:
        result = run_backtest(
            frame,
            mode=mode,
            score_column=args.score_column,
            top_k=args.top_k,
            rebalance_every=args.rebalance_every,
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
            cross_section=cross_section,
        )
        summaries.append(result.summary)
        trades.append(result.trades)
        equity_curves.append(result.equity_curve)

    summary_frame = pd.DataFrame(summaries).sort_values("total_profit", ascending=False)
    trades_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    equity_frame = pd.concat(equity_curves, ignore_index=True) if equity_curves else pd.DataFrame()

    summary_path = save_table(summary_frame, output_dir / f"{args.prefix}_summary.csv")
    trades_path = save_table(trades_frame, output_dir / f"{args.prefix}_trades.csv")
    equity_path = save_table(equity_frame, output_dir / f"{args.prefix}_equity_curve.csv")
    decile_path = save_table(deciles, output_dir / f"{args.prefix}_score_buckets.csv")
    report_path = write_report(
        summary_frame,
        cross_section,
        deciles,
        output_dir / f"{args.prefix}_investment_report.md",
        predictions=args.predictions,
        split=args.split,
        modes=args.modes,
        score_column=args.score_column,
        top_k=args.top_k,
        rebalance_every=args.rebalance_every,
        transaction_cost_bps=args.transaction_cost_bps,
        summary_path=summary_path,
        trades_path=trades_path,
        equity_path=equity_path,
        decile_path=decile_path,
    )
    logger.info(
        "Investment backtest written: summary=%s trades=%s equity=%s deciles=%s report=%s",
        summary_path,
        trades_path,
        equity_path,
        decile_path,
        report_path,
    )


@dataclass
class BacktestResult:
    summary: dict[str, object]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def run_backtest(
    frame: pd.DataFrame,
    *,
    mode: str,
    score_column: str,
    top_k: int,
    rebalance_every: int,
    initial_capital: float,
    transaction_cost_bps: float,
    cross_section: dict[str, float],
) -> BacktestResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive.")
    if frame.empty:
        raise ValueError("No prediction rows available for backtest.")

    cost_rate = transaction_cost_bps / 10_000.0
    model_name = str(frame["model_family"].iloc[0]) if "model_family" in frame.columns else "best_f6_v2"
    model_version = str(frame["model_version"].iloc[0]) if "model_version" in frame.columns else "best_f6_v2"
    rebalance_dates = sorted(frame["date"].unique())[::rebalance_every]

    equity = float(initial_capital)
    benchmark_equity = float(initial_capital)
    peak = equity
    previous_weights: dict[str, float] = {}
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    period_returns: list[float] = []
    benchmark_returns: list[float] = []

    for date in rebalance_dates:
        group = frame.loc[frame["date"] == date].sort_values(score_column, ascending=False)
        if group.empty:
            continue
        bucket_size = min(top_k, len(group))
        longs = group.head(bucket_size)
        shorts = group.tail(bucket_size)

        current_weights = _target_weights(longs, shorts, mode=mode)
        gross_return = _weighted_realized_return(group, current_weights)
        turnover = _portfolio_turnover(previous_weights, current_weights)
        transaction_cost = turnover * cost_rate
        period_return = gross_return - transaction_cost

        benchmark_return = float(group["y_true"].mean())
        equity *= 1.0 + period_return
        benchmark_equity *= 1.0 + benchmark_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        period_returns.append(period_return)
        benchmark_returns.append(benchmark_return)

        trade_row = {
            "model_family": model_name,
            "model_version": model_version,
            "date": pd.Timestamp(date),
            "mode": mode,
            "top_k": bucket_size,
            "long_symbols": ",".join(longs["symbol"].astype(str).tolist()),
            "long_realized_return": float(longs["y_true"].mean()),
            "short_symbols": ",".join(shorts["symbol"].astype(str).tolist()) if mode == "long-short" else "",
            "short_realized_return": float(shorts["y_true"].mean()) if mode == "long-short" else math.nan,
            "gross_period_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "period_return_after_cost": period_return,
            "benchmark_equal_weight_return": benchmark_return,
            "equity": equity,
            "benchmark_equity": benchmark_equity,
            "drawdown": drawdown,
        }
        trades.append(trade_row)
        equity_rows.append(
            {
                "date": pd.Timestamp(date),
                "mode": mode,
                "equity": equity,
                "benchmark_equity": benchmark_equity,
                "period_return_after_cost": period_return,
                "benchmark_equal_weight_return": benchmark_return,
                "drawdown": drawdown,
            }
        )
        previous_weights = current_weights

    returns = np.asarray(period_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows)
    total_return = equity / initial_capital - 1.0
    benchmark_total_return = benchmark_equity / initial_capital - 1.0
    periods_per_year = 252.0 / rebalance_every

    summary = {
        "model_family": model_name,
        "model_version": model_version,
        "mode": mode,
        "score_column": score_column,
        "top_k": top_k,
        "rebalance_every": rebalance_every,
        "periods": int(len(returns)),
        "initial_capital": initial_capital,
        "final_capital": equity,
        "total_profit": equity - initial_capital,
        "total_return": total_return,
        "benchmark_final_capital": benchmark_equity,
        "benchmark_total_profit": benchmark_equity - initial_capital,
        "benchmark_total_return": benchmark_total_return,
        "excess_profit_vs_benchmark": (equity - benchmark_equity),
        "excess_return_vs_benchmark": total_return - benchmark_total_return,
        "annualized_return_proxy": _annualized_return(total_return, len(returns), periods_per_year),
        "avg_period_return": float(np.mean(returns)) if len(returns) else math.nan,
        "period_volatility": float(np.std(returns, ddof=1)) if len(returns) > 1 else math.nan,
        "sharpe_proxy": _sharpe(returns, periods_per_year),
        "hit_rate": float(np.mean(returns > 0)) if len(returns) else math.nan,
        "max_drawdown": float(equity_frame["drawdown"].min()) if not equity_frame.empty else math.nan,
        "best_period_return": float(np.max(returns)) if len(returns) else math.nan,
        "worst_period_return": float(np.min(returns)) if len(returns) else math.nan,
        "mean_daily_ic": cross_section["IC"],
        "mean_daily_rankic": cross_section["RankIC"],
        "icir": cross_section["ICIR"],
        "rankicir": cross_section["RankICIR"],
        "topk_realized_return": cross_section["TopK_Return"],
        "bottomk_realized_return": cross_section["BottomK_Return"],
        "long_short_spread": cross_section["LongShort"],
    }
    return BacktestResult(summary=summary, trades=trades_frame, equity_curve=equity_frame)


def compute_cross_section_stats(frame: pd.DataFrame, *, score_column: str, top_k: int) -> dict[str, float]:
    ic_values: list[float] = []
    rankic_values: list[float] = []
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    top_hits: list[float] = []

    for _, group in frame.groupby("date"):
        clean = group.dropna(subset=["y_true", score_column])
        if len(clean) < 2:
            continue
        y_true = clean["y_true"].to_numpy(dtype=float)
        score = clean[score_column].to_numpy(dtype=float)
        ic_values.append(_safe_pearson(y_true, score))
        rankic_values.append(_safe_spearman(y_true, score))

        bucket_size = min(top_k, len(clean))
        ranked = clean.sort_values(score_column, ascending=False)
        top = ranked.head(bucket_size)
        bottom = ranked.tail(bucket_size)
        top_returns.append(float(top["y_true"].mean()))
        bottom_returns.append(float(bottom["y_true"].mean()))
        top_hits.append(float((top["y_true"] > 0).mean()))

    ic = np.asarray(ic_values, dtype=float)
    rankic = np.asarray(rankic_values, dtype=float)
    top = np.asarray(top_returns, dtype=float)
    bottom = np.asarray(bottom_returns, dtype=float)
    return {
        "IC": _nanmean(ic),
        "RankIC": _nanmean(rankic),
        "ICIR": _information_ratio(ic),
        "RankICIR": _information_ratio(rankic),
        "TopK_Return": _nanmean(top),
        "BottomK_Return": _nanmean(bottom),
        "LongShort": _nanmean(top - bottom),
        "TopK_Direction_Acc": _nanmean(np.asarray(top_hits, dtype=float)),
    }


def compute_score_bucket_returns(frame: pd.DataFrame, *, score_column: str, buckets: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in frame.groupby("date"):
        clean = group.dropna(subset=["y_true", score_column]).copy()
        if len(clean) < buckets:
            continue
        clean["score_bucket"] = pd.qcut(
            clean[score_column].rank(method="first"),
            q=buckets,
            labels=range(1, buckets + 1),
        ).astype(int)
        for bucket, bucket_frame in clean.groupby("score_bucket"):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "score_bucket": int(bucket),
                    "avg_realized_return": float(bucket_frame["y_true"].mean()),
                    "direction_acc": float((bucket_frame["y_true"] > 0).mean()),
                    "count": int(len(bucket_frame)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["score_bucket", "avg_realized_return", "direction_acc", "count"])
    daily = pd.DataFrame(rows)
    return (
        daily.groupby("score_bucket", as_index=False)
        .agg(
            avg_realized_return=("avg_realized_return", "mean"),
            direction_acc=("direction_acc", "mean"),
            avg_count=("count", "mean"),
        )
        .sort_values("score_bucket")
    )


def write_report(
    summary: pd.DataFrame,
    cross_section: dict[str, float],
    deciles: pd.DataFrame,
    output_path: Path,
    *,
    predictions: str,
    split: str,
    modes: list[str],
    score_column: str,
    top_k: int,
    rebalance_every: int,
    transaction_cost_bps: float,
    summary_path: Path,
    trades_path: Path,
    equity_path: Path,
    decile_path: Path,
) -> Path:
    ensure_dir(output_path.parent)
    table = _markdown_table(summary) if not summary.empty else "_No trades generated._"
    decile_table = _markdown_table(deciles) if not deciles.empty else "_No score buckets generated._"
    lines = [
        "# BestF6-v2 Investment Backtest",
        "",
        "This report estimates what would happen if the portfolio were built from BestF6-v2 scores.",
        "It compounds realized `target_ret_5d` on out-of-time test rows only.",
        "",
        "## Settings",
        "",
        f"- Predictions: `{predictions}`",
        f"- Split: `{split}`",
        f"- Modes: `{', '.join(modes)}`",
        f"- Score column: `{score_column}`",
        f"- Top-k: `{top_k}`",
        f"- Rebalance every: `{rebalance_every}` sessions",
        f"- Transaction cost: `{transaction_cost_bps}` bps on traded notional",
        "",
        "## Why This Supports IC",
        "",
        f"- Mean daily IC: `{cross_section['IC']:.6f}`",
        f"- Mean daily RankIC: `{cross_section['RankIC']:.6f}`",
        f"- Mean top-k realized return: `{cross_section['TopK_Return']:.6f}`",
        f"- Mean bottom-k realized return: `{cross_section['BottomK_Return']:.6f}`",
        f"- Mean long-short spread: `{cross_section['LongShort']:.6f}`",
        "",
        "If score buckets rise from low score to high score and long-short P/L is positive, "
        "the model is useful for stock ranking even when full-universe direction accuracy is modest.",
        "",
        "## P/L Summary",
        "",
        table,
        "",
        "## Score Buckets",
        "",
        decile_table,
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{summary_path.as_posix()}`",
        f"- Trades CSV: `{trades_path.as_posix()}`",
        f"- Equity curve CSV: `{equity_path.as_posix()}`",
        f"- Score buckets CSV: `{decile_path.as_posix()}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _prepare_frame(frame: pd.DataFrame, *, split: str, score_column: str) -> pd.DataFrame:
    required = {"date", "symbol", "split", "y_true", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(frame.columns)}")
    clean = frame.loc[frame["split"].astype(str) == split].copy()
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.dropna(subset=["date", "symbol", "y_true", score_column])
    return clean.sort_values(["date", "symbol"]).reset_index(drop=True)


def _target_weights(longs: pd.DataFrame, shorts: pd.DataFrame, *, mode: str) -> dict[str, float]:
    weights = {str(symbol): 1.0 / len(longs) for symbol in longs["symbol"].astype(str)}
    if mode == "long-short":
        short_weight = -1.0 / len(shorts)
        for symbol in shorts["symbol"].astype(str):
            weights[str(symbol)] = short_weight
    return weights


def _weighted_realized_return(group: pd.DataFrame, weights: dict[str, float]) -> float:
    realized_by_symbol = dict(zip(group["symbol"].astype(str), group["y_true"].astype(float), strict=False))
    return float(sum(weight * realized_by_symbol.get(symbol, 0.0) for symbol, weight in weights.items()))


def _portfolio_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    symbols = set(previous).union(current)
    return float(sum(abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols))


def _safe_pearson(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(score) == 0:
        return math.nan
    return float(np.corrcoef(y_true, score)[0, 1])


def _safe_spearman(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(score) == 0:
        return math.nan
    return float(spearmanr(y_true, score, nan_policy="omit").statistic)


def _information_ratio(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return math.nan
    std = float(np.std(clean, ddof=1))
    return float(np.mean(clean) / std) if std > 0 else math.nan


def _nanmean(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    return float(np.mean(clean)) if len(clean) else math.nan


def _annualized_return(total_return: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or total_return <= -1.0:
        return math.nan
    return float((1.0 + total_return) ** (periods_per_year / periods) - 1.0)


def _sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    if len(returns) < 2:
        return math.nan
    volatility = float(np.std(returns, ddof=1))
    if volatility <= 0:
        return math.nan
    return float(np.mean(returns) / volatility * math.sqrt(periods_per_year))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
