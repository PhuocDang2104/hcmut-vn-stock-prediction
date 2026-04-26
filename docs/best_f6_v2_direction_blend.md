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

## Upgrade Ablation

After the architecture review, two focused upgrade candidates were tested against the current production artifact:

- `Hybrid xLSTM gated direction`: gated-concat xLSTM with stronger direction training. Pairwise rank loss was disabled in this CPU rerun because full pairwise rank training was too slow locally.
- `MultiKernel CNN-BiGRU attention`: multi-kernel Conv1D branches, channel gate, BiGRU, and temporal attention pooling.

Result:

| Model | IC | RankIC | ICIR | RankICIR | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Acc | Production Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid xLSTM Direction-Excess Blend | 0.0904 | 0.0852 | 0.4653 | 0.4690 | 52.65% | 52.41% | 1.7120% | 1.7462% | 58.53% | 1.3519 |
| MultiKernel CNN-BiGRU attention | 0.0143 | -0.0006 | 0.0742 | -0.0031 | 49.99% | 50.17% | 0.9557% | 0.4391% | 56.45% | -0.6395 |
| Hybrid xLSTM gated direction | 0.0198 | 0.0275 | 0.1048 | 0.1510 | 50.52% | 50.52% | 0.5738% | 0.3471% | 52.97% | -0.7124 |

Decision:

- Keep `Hybrid xLSTM Direction-Excess Blend` as the only production candidate.
- Do not replace it with the current gated xLSTM or MultiKernel CNN-BiGRU ablations.
- Keep the upgrade metrics for audit, but loser prediction artifacts are removed from `outputs/final/model_upgrade_top5/`.
- The next worthwhile upgrade is true rank-aware training/checkpointing on stronger hardware, not another small CPU ablation.

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
| Upgrade ablation metrics | `outputs/reports/model_upgrade_top5/upgrade_metrics.csv` |
| Upgrade ablation report | `outputs/reports/model_upgrade_top5/upgrade_report.md` |

## Recommendation

Keep Hybrid xLSTM Direction-Excess Blend as the main production candidate. LightGBM-style HGBR is the strongest fast tabular baseline. CNN-LSTM is worth keeping as an auxiliary neural baseline. TCN and PatchTST are currently weaker and should not be promoted unless future tuning improves IC and LongShort5. The latest upgrade ablation did not beat the current Hybrid xLSTM production artifact, so the production prediction file is unchanged.
