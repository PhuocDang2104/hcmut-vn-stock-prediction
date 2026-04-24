from __future__ import annotations

import argparse

from vnstock.data.fetch import bootstrap_raw_data
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger
from vnstock.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw daily price data and reference files.")
    parser.add_argument("--config", default="configs/data/dataset_daily.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_fetch_raw_data")
    set_global_seed(args.seed)
    config = read_yaml(args.config)
    outputs = bootstrap_raw_data(config)
    logger.info("Raw data bootstrap complete: %s", outputs)


if __name__ == "__main__":
    main()

