# Architecture

## Layers

The repository is structured into four layers:

1. Data layer in `src/vnstock/data/` and `data/`.
2. Model layer in `src/vnstock/models/` plus one notebook per model block.
3. Shared core utilities in `src/vnstock/utils/`, `src/vnstock/evaluation/`, and `src/vnstock/visualization/`.
4. Evaluation and orchestration in `src/vnstock/pipelines/` and compare notebooks.

## Data Flow

1. Raw daily symbol files land in `data/raw/vn_stock_daily/`.
2. Ingestion merges them into an interim panel.
3. Shared dataset builder cleans, enriches, and splits the panel.
4. Model adapters export model-specific inputs from the shared panel.
5. Model notebooks or scripts train, infer, and export standardized predictions.
6. Compare utilities aggregate all prediction files into a leaderboard.

## Migration Path

Notebook sections map directly to script entry points:

- Ingest and EDA -> `run_ingest.py`
- Shared dataset builder -> `run_build_shared_dataset.py`
- Each model block -> `run_<model>.py`
- Comparison -> `run_compare.py`

This keeps notebook code thin and avoids a future rewrite when moving toward automation.

