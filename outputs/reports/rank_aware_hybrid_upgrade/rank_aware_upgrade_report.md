# Rank-Aware Hybrid Upgrade

This upgrade does not make the xLSTM backbone larger. It calibrates the existing Hybrid xLSTM score using validation-only, top-5-aware selection.

Selected rule:

```text
{"model": 0.9, "vol_neg": -0.09999999999999998}
```

Selected candidate: `model0.9_vol_neg_minus`.

Selection score on validation:

```text
0.40 * z(LongShort5) + 0.25 * z(Top5_Return)
  + 0.15 * z(Top5_Direction_Acc) + 0.10 * z(RankIC) + 0.10 * z(IC)
```

## Top Validation Candidates

| candidate | split | IC | RankIC | ICIR | RankICIR | Direction_Acc | Balanced_Acc | Top5_Return | LongShort5 | Top5_Direction_Acc | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model0.9_vol_neg_minus | valid | 0.079532 | 0.073068 | 0.275650 | 0.275096 | 0.520099 | 0.522093 | 0.007030 | 0.007717 | 0.530962 | 1.384958 |
| model0.9_excess5_plus | valid | 0.074439 | 0.069839 | 0.259541 | 0.268838 | 0.519271 | 0.519274 | 0.006535 | 0.007202 | 0.535705 | 0.926263 |
| model0.7_vol_neg_minus | valid | 0.084128 | 0.065016 | 0.280863 | 0.234871 | 0.519271 | 0.520555 | 0.007023 | 0.006656 | 0.530962 | 0.828454 |
| model0.8_mom20_plus | valid | 0.079031 | 0.068853 | 0.273407 | 0.261573 | 0.518856 | 0.520849 | 0.006285 | 0.007006 | 0.536495 | 0.809546 |
| model0.8_rs20_plus | valid | 0.079031 | 0.068853 | 0.273407 | 0.261573 | 0.518856 | 0.520849 | 0.006285 | 0.007006 | 0.536495 | 0.809546 |
| model0.7_rs20_volneg | valid | 0.075409 | 0.068573 | 0.264000 | 0.263987 | 0.516370 | 0.519770 | 0.006432 | 0.007158 | 0.533333 | 0.806059 |
| model0.9_excess5_volneg | valid | 0.073984 | 0.071329 | 0.259352 | 0.276320 | 0.519271 | 0.519274 | 0.006381 | 0.007157 | 0.532543 | 0.787153 |
| model0.8_excess5_volneg | valid | 0.070362 | 0.068034 | 0.246402 | 0.262497 | 0.514505 | 0.517984 | 0.006282 | 0.007363 | 0.534124 | 0.774758 |
| model0.9_mom20_plus | valid | 0.078115 | 0.072171 | 0.272421 | 0.276931 | 0.518856 | 0.520091 | 0.006195 | 0.006808 | 0.533333 | 0.644152 |
| model0.9_rs20_plus | valid | 0.078115 | 0.072171 | 0.272421 | 0.276931 | 0.518856 | 0.520091 | 0.006195 | 0.006808 | 0.533333 | 0.644152 |

## Locked Test Metrics

| candidate | split | IC | RankIC | ICIR | RankICIR | Direction_Acc | Balanced_Acc | Top5_Return | LongShort5 | Top5_Direction_Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_current | test | 0.090389 | 0.085171 | 0.465286 | 0.468992 | 0.525546 | 0.521865 | 0.017120 | 0.017462 | 0.585321 |
| model0.9_vol_neg_minus | test | 0.088609 | 0.079181 | 0.450379 | 0.429093 | 0.526392 | 0.520690 | 0.017140 | 0.017775 | 0.581651 |

## Cost-Adjusted Backtest

| candidate | mode | final_capital | total_profit | total_return | benchmark_total_return | excess_return_vs_benchmark | sharpe_proxy | hit_rate | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_current | long-only | 237617999.887313 | 137617999.887313 | 1.376180 | 0.255286 | 1.120894 | 2.282604 | 0.621212 | -0.222466 |
| baseline_current | long-short | 279937889.099051 | 179937889.099051 | 1.799379 | 0.255286 | 1.544093 | 2.293996 | 0.621212 | -0.209987 |
| selected_upgrade | long-only | 239413214.874944 | 139413214.874944 | 1.394132 | 0.255286 | 1.138846 | 2.369424 | 0.621212 | -0.192904 |
| selected_upgrade | long-short | 282081725.483706 | 182081725.483706 | 1.820817 | 0.255286 | 1.565531 | 2.640521 | 0.636364 | -0.110956 |

## Decision

The selected score is promoted only because it was chosen on validation and then evaluated once on out-of-time test.
It improves the portfolio-oriented metrics, especially `Top5_Return` and `LongShort5`, while `RankIC` can be slightly lower than the original baseline.

Artifacts:

- `outputs/final/rank_aware_calibrated_hybrid_predictions.parquet`
- `outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet`
- `outputs/reports/rank_aware_hybrid_upgrade/valid_candidate_metrics.csv`
- `outputs/reports/rank_aware_hybrid_upgrade/test_selected_vs_baseline_metrics.csv`
- `outputs/reports/rank_aware_hybrid_upgrade/cost_adjusted_backtest_summary.csv`
- `outputs/reports/rank_aware_hybrid_upgrade/monthly_breakdown.csv`
- `outputs/figures/rank_aware_hybrid_upgrade/test_longshort5.png`
