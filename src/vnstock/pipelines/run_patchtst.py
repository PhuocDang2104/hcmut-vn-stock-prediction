from __future__ import annotations

import argparse

from vnstock.models.patchtst.trainer import PatchTSTTrainer
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold entry point for the PatchTST block.")
    parser.add_argument("--config", default="configs/models/patchtst.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_patchtst")
    trainer = PatchTSTTrainer(read_yaml(args.config))
    artifacts = trainer.fit()
    logger.info("PatchTST scaffold manifest written to %s", artifacts.manifest_path)


if __name__ == "__main__":
    main()
