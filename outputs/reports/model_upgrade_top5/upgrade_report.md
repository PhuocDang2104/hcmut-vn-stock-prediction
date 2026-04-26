# Model Upgrade Top-5 Evaluation

Goal: test focused upgrades from the architecture note and keep only the strongest production candidate.

Candidates:

- Current `Hybrid xLSTM Direction-Excess Blend` production artifact.
- `Hybrid xLSTM gated direction`: gated-concat xLSTM with direction loss. Pairwise rank loss was disabled in this CPU run because it was too slow for a clean local rerun.
- `MultiKernel CNN-BiGRU attention`: multi-scale Conv1D, channel gate, BiGRU, attention pooling.

Selection score:

```text
0.25 * z(IC) + 0.25 * z(RankIC) + 0.25 * z(LongShort5)
  + 0.15 * z(Top5_Return) + 0.10 * z(Top5_Direction_Acc)
```

| model | rows | IC | RankIC | ICIR | RankICIR | Direction_Acc | Balanced_Acc | Top5_Return | LongShort5 | Top5_Direction_Acc | production_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid xLSTM Direction-Excess Blend | 29535 | 0.090389 | 0.085171 | 0.465286 | 0.468992 | 0.526460 | 0.524112 | 0.017120 | 0.017462 | 0.585321 | 1.351888 |
| MultiKernel CNN-BiGRU attention | 29535 | 0.014291 | -0.000560 | 0.074165 | -0.003144 | 0.499881 | 0.501688 | 0.009557 | 0.004391 | 0.564526 | -0.639504 |
| Hybrid xLSTM gated direction | 29535 | 0.019783 | 0.027479 | 0.104765 | 0.150992 | 0.505231 | 0.505250 | 0.005738 | 0.003471 | 0.529664 | -0.712385 |

Selected production candidate: `Hybrid xLSTM Direction-Excess Blend`.

If an upgrade does not beat the current Hybrid xLSTM by production score, the existing production artifact is preserved.

Artifacts:

- `outputs/reports/model_upgrade_top5/upgrade_metrics.csv`
- `outputs/final/model_upgrade_top5/`
- `outputs/figures/model_upgrade_top5/upgrade_longshort5.png`
