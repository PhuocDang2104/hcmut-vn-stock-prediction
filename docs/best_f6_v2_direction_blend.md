# Hybrid xLSTM Direction-Excess Blend

This document is the main technical note for the current production candidate.

The previous informal name was:

```text
BestF6-v2
```

The current technical name is:

```text
Hybrid xLSTM Direction-Excess Blend
```

The name is explicit: the final stock-selection score combines an xLSTM-based return/ranking signal, a direction signal, and an excess/market-relative signal.

## 1. Objective

This model is a daily K-line stock-selection model for a 95-symbol universe.

It does not try to forecast the exact next close price. The real task is:

```text
At date t, rank symbols so the top 5 names have strong realized return over t+5 sessions.
```

Main target:

```text
target_ret_5d = close[t+5] / close[t] - 1
```

Final portfolio setting:

```text
horizon = 5 sessions
top_k = 5
```

## 2. Data And Split

Canonical data source:

```text
data/processed/shared/feature_panel.parquet
```

Prediction artifact:

```text
outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet
```

Prediction rows:

| Split | Rows | Dates | Symbols | Date range |
| --- | ---: | ---: | ---: | --- |
| train | 6,161 | 1,405 | 95 | 2018-05-30 to 2023-12-29 |
| valid | 4,826 | 259 | 95 | 2024-01-02 to 2024-12-31 |
| test | 29,535 | 327 | 95 | 2025-01-02 to 2026-04-07 |

Evaluation uses only out-of-time test rows.

## 3. Input Features

The final suite uses relative, technical, volatility, volume, and market-relative features:

```text
ret_1d, ret_3d, ret_5d, ret_10d, ret_20d,
log_volume, volume_zscore_20,
hl_spread, oc_change, gap_rel,
body_ratio, upper_shadow_ratio, lower_shadow_ratio,
rolling_vol_5, rolling_vol_20, vol_ratio_5_20,
ma_ratio_5_20,
open_rel, high_rel, low_rel, ma5_rel, ma20_rel,
rsi_14, macd_hist,
market_ret_1d, market_ret_5d, market_ret_20d, market_vol_20,
excess_ret_1d, excess_ret_5d, excess_ret_20d,
relative_strength_5d, relative_strength_20d
```

Feature design principles:

- Use returns, ratios, spreads, volatility and relative-strength features instead of raw absolute prices.
- Include market-relative features because stock selection cares about outperforming the universe, not just positive absolute return.
- Keep all rolling features past-only.
- Fit symbol-level scalers on train split only.

## 4. Backbone Architecture

The underlying xLSTM-TS implementation is in:

```text
src/vnstock/models/xlstm_ts/model.py
```

The production blend uses this model family as the core sequence/ranking signal.

Input shape:

```text
[batch, context_length, num_features]
context_length = 64
```

Backbone blocks:

| Stage | Layer | Purpose |
| --- | --- | --- |
| Input projection | `Linear(num_features, hidden_dim)` | Project raw feature vector into latent state |
| Recurrent residual block | `LayerNorm -> LSTM -> Dropout -> Residual` | Learn temporal dynamics across the 64-session window |
| Feed-forward residual block | `LayerNorm -> Linear(2x) -> GELU -> Dropout -> Linear -> Residual` | Add nonlinear feature mixing after recurrent encoding |
| Output normalization | `LayerNorm(hidden_dim)` | Stabilize final representation |
| Optional pooling | `last` or `gated_concat` | Use last hidden state or learned temporal gated pooling |
| Return head | `Linear -> GELU -> Dropout -> Linear(1)` | Predict return/ranking signal |
| Direction head | `Linear -> GELU -> Dropout -> Linear(1)` | Predict up/down direction logit |

The residual LSTM block is a practical xLSTM-style proxy in this repo. It keeps the model small enough for local CPU experiments while preserving the recurrent sequence-learning bias needed for K-line data.

## 5. Blend Principle

The final score is a blend:

```text
final_score =
    0.6 * normalized(return/ranking signal)
  + 0.2 * normalized(direction signal)
  + 0.2 * normalized(excess/market-relative signal)
```

Component roles:

| Component | Role |
| --- | --- |
| Return/ranking signal | Main alpha signal for ranking symbols by expected t+5 return |
| Direction signal | Stabilizes up/down behavior and improves top-5 hit-rate |
| Excess/market-relative signal | Rewards names that outperform the universe/market context |

Why this works:

- A pure regressor can minimize error around average returns but still rank top names poorly.
- Direction-only models can improve hit-rate but may lose ranking strength.
- Excess-return signal helps separate true outperformers from broad market movement.
- The blend aligns better with top-5 stock selection than any single pointwise objective.

## 6. Metrics

Primary metrics:

| Metric | Meaning |
| --- | --- |
| `IC` | Daily cross-sectional Pearson correlation between score and realized return, averaged over dates |
| `RankIC` | Daily cross-sectional Spearman correlation between score rank and realized return rank |
| `ICIR` | Mean daily IC divided by daily IC standard deviation |
| `RankICIR` | Mean daily RankIC divided by daily RankIC standard deviation |
| `Top5_Return` | Mean realized return of the top 5 names selected each date |
| `LongShort5` | Mean top-5 return minus bottom-5 return |
| `Top5_Direction_Acc` | Fraction of selected top-5 names with positive realized return |
| `Direction_Acc` | Full-universe up/down accuracy |
| `Balanced_Acc` | Direction accuracy balanced across up/down classes |

For this project, `IC`, `RankIC`, `Top5_Return`, `LongShort5`, and `Top5_Direction_Acc` are more important than raw full-universe direction accuracy.

Reason:

```text
The production task is to rank and buy top names, not to classify every small noisy move.
```

## 7. Final Model Suite

iTransformer is removed from the final suite. Current comparison:

| Model | Scope | Rows | IC | RankIC | ICIR | RankICIR | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid xLSTM Direction-Excess Blend | full test | 29,535 | 0.0904 | 0.0852 | 0.4653 | 0.4690 | 52.65% | 52.41% | 1.7120% | 1.7462% | 58.53% |
| LightGBM-style HGBR | full test | 29,535 | 0.0545 | 0.0379 | 0.2528 | 0.1848 | 50.53% | 49.89% | 1.2315% | 1.3634% | 55.41% |
| CNN-LSTM | full test | 29,535 | 0.0522 | 0.0433 | 0.2652 | 0.2101 | 51.80% | 50.32% | 1.2969% | 1.2686% | 56.70% |
| TCN | full test | 29,535 | 0.0212 | 0.0216 | 0.1175 | 0.1205 | 48.91% | 48.89% | 0.7223% | 0.4266% | 54.68% |
| PatchTST | full test | 29,535 | 0.0185 | 0.0150 | 0.1118 | 0.0934 | 48.58% | 49.06% | 0.6983% | 0.2568% | 52.78% |
| Kronos zero-shot | partial full-test | 18,978 | 0.0069 | 0.0189 | 0.0314 | 0.1010 | 50.95% | 51.08% | 0.3581% | -0.1264% | 52.42% |

Kronos is not retrained. The current row is a partial historical CPU run covering `61/95` symbols, so it is a reference only.

## 8. Why This Model Is Kept

`Hybrid xLSTM Direction-Excess Blend` remains the production candidate because it is best on the metrics that matter most for top-5 stock selection:

- Highest `IC`: `0.0904`
- Highest `RankIC`: `0.0852`
- Highest `Top5_Return`: `1.7120%`
- Highest `LongShort5`: `1.7462%`
- Highest `Top5_Direction_Acc`: `58.53%`

The key point is not that it has the highest full-universe direction accuracy. The key point is that it ranks the cross-section better and puts stronger realized-return names into the top-5 bucket.

## 9. Leakage Controls

Controls currently enforced:

- Train, validation and test are split by time.
- Target is raw realized `target_ret_5d`.
- Feature scalers are fit on train only.
- Evaluation uses out-of-time test rows.
- Rolling features are past-only.
- Kronos is zero-shot and not fine-tuned on test.

Important caveat:

- The production prediction artifact contains train/valid/test rows, but final reported metrics use only `split == "test"`.

## 10. Final Artifacts

| Artifact | Path |
| --- | --- |
| Production prediction artifact | `outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet` |
| Canonical final suite predictions | `outputs/final/model_suite_top5/` |
| Final suite metrics | `outputs/reports/final_top5_model_suite/top5_model_suite_metrics.csv` |
| Final suite report | `outputs/reports/final_top5_model_suite/top5_model_suite_report.md` |
| Final suite figure | `outputs/figures/final_top5_model_suite/top5_model_suite_longshort.png` |
| Final suite runner | `scripts/run_final_top5_model_suite.py` |
| 10M VND investment application report | `docs/investment_application_10m.md` |

## 11. How To Reproduce

Run the final comparison suite:

```powershell
python scripts\run_final_top5_model_suite.py
```

Run tests:

```powershell
python -m pytest tests -q
```

Generate the 10M VND investment application report:

```powershell
python scripts\report_investment_application_10m.py
```

## 12. Recommendation

Keep:

```text
Hybrid xLSTM Direction-Excess Blend
```

as the current production candidate.

Use `LightGBM-style HGBR` as the strongest fast tabular baseline.

Keep `CNN-LSTM` as an auxiliary neural baseline.

Do not promote `TCN`, `PatchTST`, or partial `Kronos zero-shot` under the current metrics.
