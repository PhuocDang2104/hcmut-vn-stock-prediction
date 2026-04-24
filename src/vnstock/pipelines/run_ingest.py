from __future__ import annotations

import argparse

from vnstock.data.ingest import ingest_repo_csvs
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger
from vnstock.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw CSV files into the interim layer.")
    parser.add_argument("--config", default="configs/data/dataset_daily.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_ingest")
    set_global_seed(args.seed)
    config = read_yaml(args.config)
    outputs = ingest_repo_csvs(config)
    logger.info("Ingestion complete: %s", outputs)


if __name__ == "__main__":
    main()

