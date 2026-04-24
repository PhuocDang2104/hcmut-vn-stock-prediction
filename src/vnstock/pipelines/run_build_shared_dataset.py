from __future__ import annotations

import argparse

from vnstock.data.dataset_builder import build_all_datasets
from vnstock.data.ingest import ingest_repo_csvs
from vnstock.utils.io import load_table, read_yaml
from vnstock.utils.logging import get_logger
from vnstock.utils.paths import path_for
from vnstock.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared dataset and model-specific exports.")
    parser.add_argument("--config", default="configs/data/dataset_daily.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-interim", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_build_shared_dataset")
    set_global_seed(args.seed)
    config = read_yaml(args.config)

    if args.use_interim and (path_for("interim_root") / "cleaned_panel.parquet").exists():
        raw_panel = load_table(path_for("interim_root") / "cleaned_panel.parquet")
    else:
        ingest_repo_csvs(config)
        raw_panel = load_table(path_for("interim_root") / "cleaned_panel.parquet")

    exports = build_all_datasets(raw_panel, config)
    logger.info("Dataset build complete: %s", exports)


if __name__ == "__main__":
    main()

