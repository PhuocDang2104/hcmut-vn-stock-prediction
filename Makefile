PYTHON ?= python

.PHONY: install test fetch-raw ingest build-shared raw-eda folder-viz compare visualize-compare benchmark-2models train-xlstm-ts train-itransformer train-patchtst train-kronos

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests -q

fetch-raw:
	$(PYTHON) -m vnstock.pipelines.run_fetch_raw_data --config configs/data/dataset_daily.yaml

ingest:
	$(PYTHON) -m vnstock.pipelines.run_ingest --config configs/data/dataset_daily.yaml

build-shared:
	$(PYTHON) -m vnstock.pipelines.run_build_shared_dataset --config configs/data/dataset_daily.yaml

raw-eda:
	$(PYTHON) -m vnstock.pipelines.run_raw_eda --config configs/data/dataset_daily.yaml

folder-viz:
	$(PYTHON) -m vnstock.pipelines.run_folder_viz

compare:
	$(PYTHON) -m vnstock.pipelines.run_compare --predictions-dir outputs/predictions --split test

visualize-compare:
	$(PYTHON) -m vnstock.pipelines.run_visualize_compare --predictions-dir outputs/predictions --split test --symbol FPT

kronos-zero-shot:
	$(PYTHON) -m vnstock.pipelines.run_kronos_zero_shot --config configs/models/kronos.yaml --split test

benchmark-2models:
	$(PYTHON) -m vnstock.pipelines.run_benchmark --rebuild-dataset --use-interim --split test

train-xlstm-ts:
	$(PYTHON) -m vnstock.pipelines.run_xlstm_ts --config configs/models/xlstm_ts.yaml

train-itransformer:
	$(PYTHON) -m vnstock.pipelines.run_itransformer --config configs/models/itransformer.yaml

train-patchtst:
	$(PYTHON) -m vnstock.pipelines.run_patchtst --config configs/models/patchtst.yaml

train-kronos:
	$(PYTHON) -m vnstock.pipelines.run_kronos --config configs/models/kronos.yaml
