# BestF6-v2 Top-5 Final Evaluation

Final setting: `top_k=5`, `rebalance_every=5`, `transaction_cost=15 bps`.

## Investment P/L Proxy

| mode | final_capital | total_profit | total_return | benchmark_total_return | excess_profit_vs_benchmark | sharpe_proxy | hit_rate | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long-only | 237617999.887313 | 137617999.887313 | 1.376180 | 0.255286 | 112089411.209474 | 2.282604 | 0.621212 | -0.222466 |
| long-short | 279937889.099051 | 179937889.099051 | 1.799379 | 0.255286 | 154409300.421212 | 2.293996 | 0.621212 | -0.209987 |

## Score Buckets

| score_bucket | avg_realized_return | direction_acc | avg_count |
| --- | --- | --- | --- |
| 1.000000 | -0.000120 | 0.470457 | 18.180982 |
| 2.000000 | 0.002666 | 0.501331 | 18.122699 |
| 3.000000 | 0.004422 | 0.520317 | 17.990798 |
| 4.000000 | 0.006771 | 0.544966 | 18.122699 |
| 5.000000 | 0.012697 | 0.570646 | 18.174847 |

## Model Comparison

| model | coverage_scope | rows | dates | symbols | IC | RankIC | Direction_Acc | Balanced_Acc | Top5_Return | LongShort5 | Top5_Direction_Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BestF6-v2 | full_test | 29535 | 327 | 95 | 0.090389 | 0.085171 | 0.526460 | 0.524112 | 0.017120 | 0.017462 | 0.585321 |
| xLSTM-TS pure | full_test | 29535 | 327 | 95 | 0.058792 | 0.050748 | 0.536076 | 0.530856 | 0.013769 | 0.010794 | 0.551682 |
| iTransformer pure | full_test | 29535 | 327 | 95 | 0.019766 | 0.004707 | 0.508515 | 0.506111 | 0.010674 | 0.006089 | 0.539450 |
| Kronos zero-shot | latest_only | 95 | 2 | 95 | 0.049672 | 0.115155 | 0.610526 | 0.677198 | 0.061400 | -0.010972 | 1.000000 |

Kronos is latest-only because the current zero-shot adapter forecasts the latest test window per symbol.
It is useful as a point-in-time reference, not a full historical benchmark.

## Artifacts

- `outputs/final/best_f6_v2_predictions.parquet`
- `outputs/final/model_compare_top5/`
- `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_summary.csv`
- `outputs/reports/best_f6_v2_top5/best_f6_v2_top5_model_compare.csv`
- `outputs/figures/best_f6_v2_top5/`