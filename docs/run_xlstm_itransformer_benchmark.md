# Run xLSTM-TS And iTransformer Benchmark

This runbook covers the current production path for the two locally trained models plus
optional Kronos zero-shot inference:

- `xLSTM-TS`: train locally, infer locally, export standardized predictions.
- `iTransformer`: train locally, infer locally, export standardized predictions.
- `Kronos`: use the official pretrained foundation model zero-shot; do not finetune in this repo.

## 1. Environment

From the repo root:

```powershell
$env:PYTHONPATH='src'
python -m pip install -r requirements.txt
python -m pip install -e .
```

If the editable install is already active, `PYTHONPATH` is optional.

## 2. Build Dataset

Use the existing interim parquet to avoid re-fetching raw data:

```powershell
python -m vnstock.pipelines.run_build_shared_dataset --config configs/data/dataset_daily.yaml --use-interim
```

This produces:

- `data/processed/shared/feature_panel.parquet`
- `data/processed/shared/split_meta.json`
- `data/processed/xlstm_ts/*.npy`
- `data/processed/itransformer/*.parquet`

Current model features use relative/return-style inputs such as `open_rel`, `high_rel`, `low_rel`, `ma5_rel`, `ma20_rel`, and avoid backfilling early rolling-window rows.

## 3. Train And Infer

Run `xLSTM-TS`:

```powershell
python -m vnstock.pipelines.run_xlstm_ts --config configs/models/xlstm_ts.yaml
```

Run `iTransformer`:

```powershell
python -m vnstock.pipelines.run_itransformer --config configs/models/itransformer.yaml
```

Both local models use a multitask head:

- return head: predicts `target_ret_5d`
- direction head: predicts probability of positive `target_ret_5d`
- validation calibration: writes `outputs/predictions/<model>_calibration.json`

Each run writes:

- checkpoint and manifest under `registry/models/<model>/<run_id>/`
- predictions under `outputs/predictions/<model>_predictions.parquet`

Prediction schema:

```text
model_family,model_version,symbol,date,split,y_true,y_pred,target_name,horizon,run_id
```

## 4. Compare

```powershell
python -m vnstock.pipelines.run_compare --predictions-dir outputs/predictions --leaderboard-output outputs/metrics/leaderboard_xlstm_itransformer.csv --combined-output outputs/predictions/all_predictions_xlstm_itransformer.parquet --split test
```

Primary metrics:

- Lower is better: `huber`, `mae`, `rmse`
- Higher is better: `information_coefficient`, `directional_accuracy`, `top_k_realized_return`, `long_short_spread`

Latest local result on the `test` split:

| model_family | rows | huber | mae | rmse | information_coefficient | directional_accuracy | top_k_realized_return | long_short_spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| xlstm_ts | 29535 | 0.001101 | 0.039319 | 0.056310 | 0.081936 | 0.528424 | 0.015342 | 0.012236 |
| itransformer | 29535 | 0.001123 | 0.039838 | 0.056602 | 0.010131 | 0.507974 | 0.013200 | 0.011400 |

## 5. Visualize

```powershell
python -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT
```

Outputs:

- `outputs/reports/model_compare/benchmark_summary.md`
- `outputs/reports/model_compare/leaderboard.csv`
- `outputs/figures/model_compare/leaderboard_huber.png`
- `outputs/figures/model_compare/leaderboard_ic.png`
- `outputs/figures/model_compare/prediction_scatter.png`
- `outputs/figures/model_compare/residual_distribution.png`
- `outputs/figures/model_compare/daily_ic_rolling20.png`
- `outputs/figures/model_compare/fpt_actual_vs_pred.png`

## 6. One Command For The Local Two-Model Suite

```powershell
python -m vnstock.pipelines.run_benchmark --rebuild-dataset --use-interim --split test
python -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT
```

`run_benchmark` intentionally defaults to only `xlstm_ts` and `itransformer`.

## 7. Kronos Zero-Shot Path

Kronos should be run as the official pretrained foundation model, not through the local trainer.

Official reference:

- GitHub: `https://github.com/shiyu-coder/Kronos`
- Pretrained models: `NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-base`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`

Install or clone the official repo once:

```powershell
git clone --depth 1 https://github.com/shiyu-coder/Kronos external/Kronos
python -m pip install einops==0.8.1 safetensors==0.6.2
```

Run latest test-window zero-shot inference for all symbols:

```powershell
python -m vnstock.pipelines.run_kronos_zero_shot --config configs/models/kronos.yaml --kronos-repo external/Kronos --split test --device cpu
```

Then compare all 3 models on the common `(symbol, date)` intersection:

```powershell
python -m vnstock.pipelines.run_compare --predictions-dir outputs/predictions --leaderboard-output outputs/metrics/leaderboard_3models_aligned.csv --combined-output outputs/predictions/all_predictions_3models.parquet --split test --align-intersection
python -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT --align-intersection
```

Latest aligned result is based on `95` rows, one latest test point per symbol from Kronos:

| model_family | rows | huber | mae | rmse | information_coefficient | directional_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| itransformer | 95 | 0.001449 | 0.048728 | 0.062051 | 0.198601 | 0.778947 |
| xlstm_ts | 95 | 0.001580 | 0.052160 | 0.065577 | 0.150643 | 0.526316 |
| kronos | 95 | 0.001977 | 0.060138 | 0.079780 | 0.037157 | 0.536842 |

Expected direct API pattern:

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=5,
    T=1.0,
    top_p=0.9,
    sample_count=1,
)
```

Then convert the predicted close at `t+5` to `target_ret_5d` and export the same prediction contract as the other models.

This repo avoids local Kronos finetuning by default.
