# Notebook Conventions

Each model notebook follows the same eight sections:

1. Setup
2. Load config
3. Load data
4. Prepare model input
5. Train
6. Infer
7. Evaluate and visualize
8. Export artifacts

## Rules

- Keep reusable logic in `src/vnstock/`, not inside notebook cells.
- Read only from `data/processed/shared/` or the model-specific processed directory.
- Export predictions and metrics using the standardized file names in `outputs/`.
- Use the model config under `configs/models/`.

