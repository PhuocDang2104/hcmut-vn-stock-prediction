from __future__ import annotations

import argparse

from vnstock.data.folder_viz import visualize_raw_data_folders
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts for each folder under data/raw.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    logger = get_logger("vnstock.pipelines.run_folder_viz")
    outputs = visualize_raw_data_folders()
    logger.info("Folder visualization outputs written: %s", outputs)


if __name__ == "__main__":
    main()

