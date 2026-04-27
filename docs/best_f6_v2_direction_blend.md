# Hybrid xLSTM Direction-Excess Blend

## Rename

The previous informal name `BestF6-v2` is replaced by the more technical name:

```text
Hybrid xLSTM Direction-Excess Blend
```

This name is more explicit because the final score blends:

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

## Principle

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

## Final Model Suite

iTransformer is removed from the final suite. Current comparison:

| Model | Scope | Rows | IC | RankIC | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid xLSTM Direction-Excess Blend | full test | 29,535 | 0.0904 | 0.0852 | 52.65% | 52.41% | 1.7120% | 1.7462% | 58.53% |
| LightGBM-style HGBR | full test | 29,535 | 0.0545 | 0.0379 | 50.53% | 49.89% | 1.2315% | 1.3634% | 55.41% |
| CNN-LSTM | full test | 29,535 | 0.0522 | 0.0433 | 51.80% | 50.32% | 1.2969% | 1.2686% | 56.70% |
| TCN | full test | 29,535 | 0.0212 | 0.0216 | 48.91% | 48.89% | 0.7223% | 0.4266% | 54.68% |
| PatchTST | full test | 29,535 | 0.0185 | 0.0150 | 48.58% | 49.06% | 0.6983% | 0.2568% | 52.78% |
| Kronos zero-shot | partial full-test | 18,978 | 0.0069 | 0.0189 | 50.95% | 51.08% | 0.3581% | -0.1264% | 52.42% |

Kronos is not retrained. The current row is partial because the CPU full-test zero-shot process stopped at `61/95` symbols.

## Leakage Controls

- Train, validation and test split by time.
- Target is raw realized return.
- Scalers are fit from train only.
- Top-5 metrics use out-of-time test rows.
- Kronos is treated as zero-shot reference, not fine-tuned.

## Final Artifacts

| Artifact | Path |
| --- | --- |
| Hybrid xLSTM final prediction | `outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet` |
| Final suite predictions | `outputs/final/model_suite_top5/` |
| Final suite metrics | `outputs/reports/final_top5_model_suite/top5_model_suite_metrics.csv` |
| Final suite report | `outputs/reports/final_top5_model_suite/top5_model_suite_report.md` |
| Final suite figure | `outputs/figures/final_top5_model_suite/top5_model_suite_longshort.png` |

## Recommendation

Keep Hybrid xLSTM Direction-Excess Blend as the main production candidate. LightGBM-style HGBR is the strongest fast tabular baseline. CNN-LSTM is worth keeping as an auxiliary neural baseline. TCN and PatchTST are currently weaker and should not be promoted unless future tuning improves IC and LongShort5.
