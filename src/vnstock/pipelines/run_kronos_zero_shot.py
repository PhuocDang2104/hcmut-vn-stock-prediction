from __future__ import annotations

import argparse

from vnstock.models.kronos.zero_shot import run_kronos_zero_shot
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Kronos foundation model zero-shot inference.")
    parser.add_argument("--config", default="configs/models/kronos.yaml")
    parser.add_argument("--kronos-repo", default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_kronos_zero_shot")
    config = read_yaml(args.config)
    if args.kronos_repo:
        config["kronos_repo"] = args.kronos_repo
    if args.max_symbols is not None:
        config["max_symbols"] = args.max_symbols
    if args.split:
        config["split"] = args.split
    if args.device:
        config["device"] = args.device
    predictions_path = run_kronos_zero_shot(config)
    logger.info("Kronos zero-shot predictions written to %s", predictions_path)


if __name__ == "__main__":
    main()
