# Architecture Ablation: A1 Multi-Head LongOnlyRank

Goal: test an architecture-only upgrade while keeping the Hybrid xLSTM baseline as the production reference.

A1 changes:

- Same residual xLSTM-style backbone size: hidden `96`, `2` blocks, dropout `0.1`.
- Multi-head outputs: `ret_1d`, `ret_3d`, `ret_5d`, computed `ret_10d`, `excess_5d`, `rank_5d`, `dir_5d`.
- Final long-only score: `0.50*z(rank_5d) + 0.25*z(ret_5d) + 0.15*z(excess_5d) + 0.10*z(dir_5d)`.
- Checkpointing uses validation long-only score, not validation loss.

Promotion rule:

- `total_return >= baseline`
- `Top5_Return >= baseline`
- `Top5_Direction_Acc` not worse by more than `0.2pp`
- `RankIC` not worse by more than `0.005`

## Locked Test Metrics

| model | rows | IC | RankIC | Top5_Return | Top5_Direction_Acc | total_return | final_capital | sharpe_proxy | max_drawdown | promote |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid baseline | 28585 | 0.090398 | 0.087056 | 0.016467 | 0.582965 | 1.183557 | 218355690.183120 | 2.118655 | -0.222466 | False |
| A1 Multi-Head LongOnlyRank | 28585 | 0.041875 | 0.044454 | 0.009171 | 0.550789 | 0.421957 | 142195732.951659 | 1.150479 | -0.233440 | False |

## Training History

| epoch | train_loss | valid_selection_score | valid_IC | valid_RankIC | valid_Top5_Return | valid_Top5_Direction_Acc | valid_total_return | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 0.076503 | 0.172809 | 0.030256 | 0.035179 | 0.005538 | 0.533718 | 0.273798 | 188.652785 |
| 2.000000 | 0.074812 | 0.157534 | 0.027717 | 0.026245 | 0.004838 | 0.531410 | 0.149671 | 134.479333 |
| 3.000000 | 0.073504 | 0.171877 | 0.045572 | 0.051740 | 0.005287 | 0.544487 | 0.189989 | 136.455477 |

## Decision

Promote A1 only if the `promote` column is true. Otherwise keep `Hybrid xLSTM Direction-Excess Blend` as production.

Artifacts:

- `outputs/reports/architecture_ablation_longonly/architecture_ablation_metrics.csv`
- `outputs/reports/architecture_ablation_longonly/training_history.csv`
- `outputs/final/architecture_ablation_longonly/a1_multihead_longonlyrank_predictions.parquet`
- `outputs/final/architecture_ablation_longonly/baseline_same_rows_predictions.parquet`
- `outputs/figures/architecture_ablation_longonly/test_total_return.png`