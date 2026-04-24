# Model Blocks

## xLSTM-TS

- Input: dense tensor `[num_samples, context_length, num_features]`
- Export path: `data/processed/xlstm_ts/`
- Notebook: `notebooks/10_xlstm_ts_block.ipynb`

## iTransformer

- Input: parquet panel plus sequence metadata
- Export path: `data/processed/itransformer/`
- Notebook: `notebooks/11_itransformer_block.ipynb`

## PatchTST

- Input: parquet panel plus patching metadata
- Export path: `data/processed/patchtst/`
- Notebook: `notebooks/12_patchtst_block.ipynb`

## Kronos

- Input: adapter-managed CSV export isolated from the shared builder internals
- Export path: `data/processed/kronos/`
- Notebook: `notebooks/13_kronos_block.ipynb`
