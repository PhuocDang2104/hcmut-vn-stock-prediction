from __future__ import annotations

import argparse

from vnstock.models.xlstm_ts.trainer import XLSTMTrainer
from vnstock.utils.io import read_yaml
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold entry point for the xLSTM-TS block.")
    parser.add_argument("--config", default="configs/models/xlstm_ts.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_xlstm_ts")
    trainer = XLSTMTrainer(read_yaml(args.config))
    artifacts = trainer.fit()
    logger.info(
        "xLSTM-TS run complete: manifest=%s predictions=%s",
        artifacts.manifest_path,
        artifacts.details.get("predictions_path"),
    )


if __name__ == "__main__":
    main()
