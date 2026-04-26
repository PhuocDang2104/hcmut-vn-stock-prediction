# BestF6-v2 Top-5 Direction/Excess Blend

## Objective

BestF6-v2 is the final kept flow for the current stock-selection architecture. The model does not answer "what is tomorrow's close price?". It answers:

```text
Which stocks should be ranked highest for realized return over the next 5 sessions?
```

Final target:

```text
target_ret_5d = close[t+5] / close[t] - 1
```

Final trading metric uses `top_k=5` only.

## Principle

BestF6-v2 blends three signals:

```text
final_score =
    0.6 * normalized(return/ranking signal)
  + 0.2 * normalized(direction signal)
  + 0.2 * normalized(excess/market-relative signal)
```

Signal meaning:

- Return/ranking signal ranks stocks by expected 5-session return.
- Direction signal stabilizes the up/down decision.
- Excess/market-relative signal prefers stocks that outperform the market/universe, not just stocks moving with the broad market.

The model is selected for stock ranking, so IC, RankIC, Top-5 Return and Long-short Spread matter more than raw full-universe Direction Accuracy.

## Leakage Controls

- Train/valid/test are split by time.
- `y_true` is always raw realized return, not denoised return.
- Feature scaling is fit from train only.
- Backtest uses test split only.
- Rebalance interval is 5 sessions to align with horizon t+5.
- Future `close[t+5]` is never used in input features.

## Final Top-5 Metrics

| Model | Scope | Rows | IC | RankIC | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Direction Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BestF6-v2 | full test | 29,535 | 0.0904 | 0.0852 | 52.65% | 52.41% | 1.7120% | 1.7462% | 58.53% |
| xLSTM-TS pure | full test | 29,535 | 0.0588 | 0.0507 | 53.61% | 53.09% | 1.3769% | 1.0794% | 55.17% |
| iTransformer pure | full test | 29,535 | 0.0198 | 0.0047 | 50.85% | 50.61% | 1.0674% | 0.6089% | 53.94% |
| Kronos zero-shot | latest only | 95 | 0.0497 | 0.1152 | 61.05% | 67.72% | 6.1400% | -1.0972% | 100.00% |

Kronos is latest-only in the current adapter, so it is not directly comparable with full-test models. It is kept as a point-in-time foundation-model reference.

Decision:

- xLSTM-TS pure has slightly higher raw Direction Accuracy.
- BestF6-v2 wins the stock-selection metrics that matter for this project: IC, RankIC, Top-5 Return and LongShort5.
- iTransformer is weaker on this run.
- Kronos needs historical zero-shot inference before it can be judged fairly.

## Investment P/L Proxy

Command:

```powershell
python scripts\evaluate_best_f6_v2_top5.py
```

Backtest assumptions:

- Uses only test split.
- Selects top 5 stocks by BestF6-v2 score.
- Rebalances every 5 sessions.
- Initial capital is 100,000,000.
- Transaction cost is 15 bps on traded notional.
- This is a return-compounding proxy, not a broker-level simulation with liquidity, tax, lot size, and slippage.

| Mode | Final Capital | Profit | Total Return | Benchmark Return | Excess Profit | Sharpe Proxy | Hit Rate | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Long-only top-5 | 237,617,999 | 137,617,999 | 137.62% | 25.53% | 112,089,411 | 2.2826 | 62.12% | -22.25% |
| Long-short top/bottom-5 | 279,937,889 | 179,937,889 | 179.94% | 25.53% | 154,409,300 | 2.2940 | 62.12% | -21.00% |

## IC Evidence

Score buckets are monotonic on test:

| Score Bucket | Avg Realized t+5 Return | Direction Acc |
| --- | ---: | ---: |
| Lowest 20% | -0.0120% | 47.05% |
| 2 | 0.2666% | 50.13% |
| 3 | 0.4422% | 52.03% |
| 4 | 0.6771% | 54.50% |
| Highest 20% | 1.2697% | 57.06% |

This is the clearest evidence that the score has ranking signal: higher score buckets realize higher future returns.

## Visual Outputs

| Figure | Path |
| --- | --- |
| Equity curve | `outputs/figures/best_f6_v2_top5/best_f6_v2_top5_equity_curve.png` |
| Score buckets | `outputs/figures/best_f6_v2_top5/best_f6_v2_top5_score_buckets.png` |
| Model LongShort5 comparison | `outputs/figures/best_f6_v2_top5/best_f6_v2_model_longshort5.png` |

## Artifacts

| Artifact | Path |
| --- | --- |
| Final prediction | `outputs/final/best_f6_v2_predictions.parquet` |
| Model comparison predictions | `outputs/final/model_compare_top5/` |
| Top-5 summary | `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_summary.csv` |
| Model comparison CSV | `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_model_compare.csv` |
| Trade log | `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_trades.csv` |
| Equity curve CSV | `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_equity_curve.csv` |
| Final report | `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_report.md` |

## Final Recommendation

Keep BestF6-v2 top-5 as the main production candidate for stock selection. Do not optimize only for full-universe Direction Accuracy; for this task, ranking quality and top-k realized return are more aligned with the investment objective.
