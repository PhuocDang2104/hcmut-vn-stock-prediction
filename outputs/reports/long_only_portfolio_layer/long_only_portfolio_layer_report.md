# Hybrid xLSTM Long-Only Portfolio Layer

Goal: keep the Hybrid xLSTM baseline as the core ranking signal and test only long-only portfolio-layer improvements.

Selection is done on validation using long-only after-cost metrics. `LongShort5` is diagnostic only and is not used as the main selection objective.

Selected configuration:

- Score spec: `baseline_rs20_010` with weights `{"base_score": 0.9, "relative_strength_20d": 0.1}`
- Portfolio spec: `confidence_cash_50`

Selection score:

```text
0.30 * z(total_return) + 0.20 * z(sharpe_proxy)
  - 0.15 * z(abs(max_drawdown)) - 0.10 * z(avg_turnover)
  + 0.15 * z(Top5_Direction_Acc) + 0.10 * z(RankIC)
```

## Top Validation Candidates

| score_spec | portfolio_spec | IC | RankIC | Top5_Return | Top5_Direction_Acc | total_return | sharpe_proxy | max_drawdown | avg_turnover | avg_exposure | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_rs20_010 | confidence_cash_50 | 0.078115 | 0.072171 | 0.006195 | 0.533333 | 0.357054 | 1.472853 | -0.103774 | 1.648077 | 0.884615 | 1.412134 |
| baseline_excess5_005 | confidence_cash_50 | 0.075556 | 0.072718 | 0.006557 | 0.532543 | 0.374003 | 1.531654 | -0.118590 | 1.690385 | 0.903846 | 1.355320 |
| baseline_rs20_010 | filter_liquidity | 0.078115 | 0.072171 | 0.006195 | 0.533333 | 0.355427 | 1.540874 | -0.103229 | 1.780769 | 1.000000 | 1.210273 |
| baseline_rs20_005 | confidence_cash_50 | 0.077312 | 0.074206 | 0.006280 | 0.530171 | 0.336048 | 1.405498 | -0.122098 | 1.686538 | 0.913462 | 1.152607 |
| baseline_excess5_005 | filter_liquidity | 0.075556 | 0.072718 | 0.006557 | 0.532543 | 0.344454 | 1.494401 | -0.118056 | 1.788462 | 1.000000 | 1.056492 |
| baseline_rs20_010 | rank_weighted | 0.078115 | 0.072171 | 0.006195 | 0.533333 | 0.361999 | 1.421427 | -0.111558 | 1.841154 | 1.000000 | 1.000792 |
| baseline_excess5_005 | rank_weighted | 0.075556 | 0.072718 | 0.006557 | 0.532543 | 0.360090 | 1.426505 | -0.124001 | 1.843462 | 1.000000 | 0.932999 |
| baseline_rs20_005 | rank_weighted | 0.077312 | 0.074206 | 0.006280 | 0.530171 | 0.356214 | 1.421955 | -0.126814 | 1.836154 | 1.000000 | 0.931530 |
| baseline_rs20_005 | filter_liquidity | 0.077312 | 0.074206 | 0.006280 | 0.530171 | 0.315526 | 1.394610 | -0.121564 | 1.780769 | 1.000000 | 0.907731 |
| baseline_only | confidence_cash_50 | 0.076360 | 0.073134 | 0.006060 | 0.526219 | 0.331504 | 1.392675 | -0.118590 | 1.705769 | 0.923077 | 0.898762 |
| baseline_excess5_010 | confidence_cash_50 | 0.074439 | 0.069839 | 0.006535 | 0.535705 | 0.279635 | 1.247434 | -0.122098 | 1.617308 | 0.875000 | 0.889498 |
| baseline_rs20_downside | confidence_cash_50 | 0.074876 | 0.070757 | 0.006170 | 0.528590 | 0.330703 | 1.386742 | -0.122098 | 1.686538 | 0.913462 | 0.830243 |

## Locked Test Comparison

| candidate | score_spec | portfolio_spec | IC | RankIC | Top5_Return | Top5_Direction_Acc | total_return | final_capital | sharpe_proxy | max_drawdown | avg_turnover | avg_exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_current | baseline_only | equal_top5 | 0.090389 | 0.085171 | 0.017120 | 0.585321 | 1.376180 | 237617999.887313 | 2.282604 | -0.222466 | 1.500000 | 1.000000 |
| selected_long_only | baseline_rs20_010 | confidence_cash_50 | 0.090521 | 0.079418 | 0.018580 | 0.590214 | 1.252804 | 225280447.464632 | 2.343011 | -0.175148 | 1.224242 | 0.810606 |

## Decision

Production decision: `baseline_current`.

Promotion rule: the selected candidate must improve locked-test long-only total return and keep Top5 Direction Accuracy approximately unchanged. Otherwise baseline remains production and the selected candidate is only a defensive overlay idea.

Artifacts:

- `outputs/final/hybrid_xlstm_baseline_predictions.parquet`
- `outputs/final/hybrid_xlstm_long_only_production_predictions.parquet`
- `outputs/reports/long_only_portfolio_layer/valid_long_only_grid.csv`
- `outputs/reports/long_only_portfolio_layer/test_selected_vs_baseline.csv`
- `outputs/reports/long_only_portfolio_layer/selected_long_only_config.json`
- `outputs/figures/long_only_portfolio_layer/test_total_return.png`