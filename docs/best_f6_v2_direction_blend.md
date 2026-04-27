# Rank-Aware Calibrated Hybrid xLSTM

## Current Production Name

The previous informal name `BestF6-v2` was first replaced by:

```text
Hybrid xLSTM Direction-Excess Blend
```

The current production artifact is now:

```text
Rank-Aware Calibrated Hybrid xLSTM
```

This model keeps the Hybrid xLSTM score as the core signal, then applies validation-selected top-5 calibration.

The original Hybrid name is still useful because the base score blends:

- an xLSTM return/ranking signal,
- a direction signal,
- an excess/market-relative signal.

## Objective

The model is a 5-session stock-selection model. It does not forecast exact next close price. It ranks symbols by expected realized return over the next 5 sessions.

Target:

```text
target_ret_5d = close[t+5] / close[t] - 1
```

Final portfolio metric uses `top_k=5`.

## Original Hybrid Principle

```text
final_score =
    0.6 * normalized(return/ranking signal)
  + 0.2 * normalized(direction signal)
  + 0.2 * normalized(excess/market-relative signal)
```

Why this works better than a pure regressor:

- The return/ranking component optimizes stock ordering.
- The direction component improves up/down stability.
- The excess-return component helps select stocks that outperform the market/universe.

## Rank-Aware Calibration

The latest upgrade does not make the xLSTM backbone larger. It changes scoring to better match the true portfolio objective: choose top 5 names each day.

Validation-selected rule:

```text
rank_aware_score =
    0.9 * cross_section_zscore(previous_hybrid_score)
  - 0.1 * cross_section_zscore(vol_neg)

where vol_neg = -rolling_vol_20
```

Equivalent interpretation:

```text
rank_aware_score =
    0.9 * z(previous_hybrid_score)
  + 0.1 * z(rolling_vol_20)
```

The small volatility-regime tilt was selected on validation because it improved top-5/long-short metrics. It uses only information available at date `t`.

## Final Model Suite

iTransformer is removed from the final suite. Current comparison:

| Model | Scope | Rows | IC | RankIC | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank-Aware Calibrated Hybrid xLSTM | full test | 29,535 | 0.0886 | 0.0792 | 52.64% | 52.07% | 1.7140% | 1.7775% | 58.17% |
| Hybrid xLSTM Direction-Excess Blend baseline | full test | 29,535 | 0.0904 | 0.0852 | 52.55% | 52.19% | 1.7120% | 1.7462% | 58.53% |
| LightGBM-style HGBR | full test | 29,535 | 0.0545 | 0.0379 | 50.53% | 49.89% | 1.2315% | 1.3634% | 55.41% |
| CNN-LSTM | full test | 29,535 | 0.0522 | 0.0433 | 51.80% | 50.32% | 1.2969% | 1.2686% | 56.70% |
| TCN | full test | 29,535 | 0.0212 | 0.0216 | 48.91% | 48.89% | 0.7223% | 0.4266% | 54.68% |
| PatchTST | full test | 29,535 | 0.0185 | 0.0150 | 48.58% | 49.06% | 0.6983% | 0.2568% | 52.78% |
| Kronos zero-shot | partial full-test | 18,978 | 0.0069 | 0.0189 | 50.95% | 51.08% | 0.3581% | -0.1264% | 52.42% |

Kronos is not retrained. The current row is partial because the CPU full-test zero-shot process stopped at `61/95` symbols.

## Rank-Aware Upgrade Result

The selected upgrade was chosen on validation and evaluated once on test.

| Candidate | IC | RankIC | ICIR | RankICIR | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 0.0904 | 0.0852 | 0.4653 | 0.4690 | 52.55% | 52.19% | 1.7120% | 1.7462% | 58.53% |
| Rank-aware calibrated | 0.0886 | 0.0792 | 0.4504 | 0.4291 | 52.64% | 52.07% | 1.7140% | 1.7775% | 58.17% |

Cost-adjusted long-short backtest with `top_k=5`, rebalance every `5` sessions and `15 bps` one-way cost:

| Candidate | Final capital | Total return | Sharpe proxy | Hit rate | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 279,937,889 | 179.94% | 2.2940 | 62.12% | -21.00% |
| Rank-aware calibrated | 282,081,725 | 182.08% | 2.6405 | 63.64% | -11.10% |

Decision: promote `Rank-Aware Calibrated Hybrid xLSTM` because it improves portfolio-oriented metrics and risk-adjusted backtest behavior. The trade-off is slightly lower `RankIC`.

## Leakage Controls

- Train, validation and test split by time.
- Target is raw realized return.
- Scalers are fit from train only.
- Top-5 metrics use out-of-time test rows.
- Kronos is treated as zero-shot reference, not fine-tuned.

## Final Artifacts

| Artifact | Path |
| --- | --- |
| Current production prediction | `outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet` |
| Rank-aware prediction snapshot | `outputs/final/rank_aware_calibrated_hybrid_predictions.parquet` |
| Final suite predictions | `outputs/final/model_suite_top5/` |
| Final suite metrics | `outputs/reports/final_top5_model_suite/top5_model_suite_metrics.csv` |
| Final suite report | `outputs/reports/final_top5_model_suite/top5_model_suite_report.md` |
| Final suite figure | `outputs/figures/final_top5_model_suite/top5_model_suite_longshort.png` |
| Rank-aware upgrade report | `outputs/reports/rank_aware_hybrid_upgrade/rank_aware_upgrade_report.md` |

## Recommendation

Keep Rank-Aware Calibrated Hybrid xLSTM as the main production candidate. Keep Hybrid xLSTM Direction-Excess Blend as the baseline source score. LightGBM-style HGBR remains the strongest fast tabular baseline. CNN-LSTM is worth keeping as an auxiliary neural baseline. TCN and PatchTST are currently weaker and should not be promoted unless future tuning improves IC and LongShort5.
