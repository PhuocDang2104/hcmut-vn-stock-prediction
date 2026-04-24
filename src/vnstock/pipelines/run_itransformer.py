from __future__ import annotations

import argparse

from vnstock.models.itransformer.trainer import ITransformerTrainer
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold entry point for the iTransformer block.")
    parser.add_argument("--config", default="configs/models/itransformer.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_itransformer")
    trainer = ITransformerTrainer(read_yaml(args.config))
    artifacts = trainer.fit()
    logger.info(
        "iTransformer run complete: manifest=%s predictions=%s",
        artifacts.manifest_path,
        artifacts.details.get("predictions_path"),
    )


if __name__ == "__main__":
    main()
