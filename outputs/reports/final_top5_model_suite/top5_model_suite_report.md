# Final Top-5 Model Suite

iTransformer is intentionally removed from this final suite.
Kronos is loaded as the current zero-shot reference and is not retrained.

| model | coverage_scope | rows | IC | RankIC | Direction_Acc | Balanced_Acc | Top5_Return | LongShort5 | Top5_Direction_Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid xLSTM Direction-Excess Blend | full_test | 29535 | 0.090389 | 0.085171 | 0.526460 | 0.524112 | 0.017120 | 0.017462 | 0.585321 |
| LightGBM-style HGBR | full_test | 29535 | 0.054475 | 0.037861 | 0.505299 | 0.498945 | 0.012315 | 0.013634 | 0.554128 |
| CNN-LSTM | full_test | 29535 | 0.052248 | 0.043325 | 0.517996 | 0.503177 | 0.012969 | 0.012686 | 0.566972 |
| TCN | full_test | 29535 | 0.021221 | 0.021601 | 0.489081 | 0.488898 | 0.007223 | 0.004266 | 0.546789 |
| PatchTST | full_test | 29535 | 0.018508 | 0.014950 | 0.485763 | 0.490550 | 0.006983 | 0.002568 | 0.527829 |
| Kronos zero-shot | full_test | 18978 | 0.006897 | 0.018904 | 0.509537 | 0.510822 | 0.003581 | -0.001264 | 0.524159 |

## Artifacts

- `outputs/reports/final_top5_model_suite/top5_model_suite_metrics.csv`
- `outputs/final/model_suite_top5/`
- `outputs/figures/final_top5_model_suite/top5_model_suite_longshort.png`