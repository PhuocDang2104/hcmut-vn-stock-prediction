# VN Stock Research Repo

Research-first repository for Vietnamese equity forecasting with one shared data source, four model-specific notebooks, and a standardized prediction contract for fair comparison.

## Design Goals

- One canonical dataset drives every model block.
- Each model keeps its own notebook and adapter without duplicating ingestion logic.
- Outputs are normalized into one prediction schema for comparison and ranking.
- Notebook workflows map cleanly to `src/vnstock/pipelines/` scripts later.

## Repository Flow

1. `notebooks/01_data_ingest_and_eda.ipynb`
2. `notebooks/02_shared_dataset_builder.ipynb`
3. `notebooks/10_xlstm_ts_block.ipynb`
4. `notebooks/11_itransformer_block.ipynb`
5. `notebooks/12_patchtst_block.ipynb`
6. `notebooks/13_kronos_block.ipynb`
7. `notebooks/20_compare_models.ipynb`
8. `notebooks/21_final_decision_support.ipynb`

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Build the shared dataset:

```bash
python -m vnstock.pipelines.run_build_shared_dataset --config configs/data/dataset_daily.yaml
```

Compare exported predictions:

```bash
python -m vnstock.pipelines.run_compare --predictions-dir outputs/predictions --split test
```

Run the current local two-model benchmark and visualization:

```bash
python -m vnstock.pipelines.run_benchmark --rebuild-dataset --use-interim --split test
python -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT
```

## Key Contracts

- Raw schema: `symbol,date,open,high,low,close,volume,value,source`
- Shared panel: see `docs/data_contract.md`
- Standard prediction output: `model_family,model_version,symbol,date,split,y_true,y_pred,target_name,horizon,run_id`

## Core Layout

- `configs/`: project, data, model, and evaluation configuration.
- `data/`: raw, interim, processed shared data, and model-specific exports.
- `notebooks/`: EDA, shared builder, model blocks, and comparison notebooks.
- `src/vnstock/`: reusable code for data, model wrappers, evaluation, visualization, and pipelines.
- `outputs/`: predictions, metrics, figures, and reports.
- `registry/`: checkpoints and experiment manifests.
- `docs/`: architecture, data contracts, notebook conventions.

## Notes

- Current model roster: `xLSTM-TS`, `iTransformer`, `PatchTST`, `Kronos`.
- `xLSTM-TS` and `iTransformer` now have local train/infer/export flows.
- `Kronos` should use the official pretrained foundation model zero-shot; do not finetune it in the local benchmark path.
- Empty directories are retained with `.gitkeep` where useful.

See `docs/run_xlstm_itransformer_benchmark.md` for the exact benchmark commands and output paths.
