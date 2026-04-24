from __future__ import annotations

import argparse

from vnstock.data.raw_eda import run_raw_eda
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first-pass raw data EDA figures and reports.")
    parser.add_argument("--config", default="configs/data/dataset_daily.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_raw_eda")
    config = read_yaml(args.config)
    outputs = run_raw_eda(config)
    logger.info("Raw EDA outputs written: %s", outputs)


if __name__ == "__main__":
    main()

