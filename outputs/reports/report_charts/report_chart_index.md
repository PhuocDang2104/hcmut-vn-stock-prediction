# Report Chart Pack

Figure directory: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/figures/report_charts`

| Figure | Purpose |
| --- | --- |
| `01_data_split_timeline.png` | Chronological train, validation, and test split. |
| `02_model_comparison_metrics.png` | IC, RankIC, Top5 Return, and LongShort5 across models. |
| `03_top5_return_vs_longshort.png` | Portfolio-level model trade-off between top5 return and long-short spread. |
| `04_daily_ic_rolling_ic.png` | Daily IC plus rolling IC/RankIC stability over time. |
| `05_score_bucket_returns.png` | Monotonic relationship between score bucket and realized return. |
| `06_equity_curve_10m.png` | 10M VND long-only top-5 equity curve vs benchmark. |
| `07_monthly_top5_stability.png` | Monthly top5 return and IC stability for paper/report discussion. |
| `08_drawdown_curve_10m.png` | Portfolio drawdown path for risk discussion. |
| `09_prediction_score_distribution.png` | Distribution of model scores on the test set. |

Generated data tables:

- `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/report_charts/daily_hybrid_metrics.csv`
- `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/report_charts/monthly_hybrid_metrics.csv`

Regenerate:

```powershell
python scripts\generate_report_charts.py
```
