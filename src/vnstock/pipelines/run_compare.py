from __future__ import annotations

import argparse

from vnstock.evaluation.compare import export_all_predictions
from vnstock.evaluation.leaderboard import export_leaderboard
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate prediction files into compare artifacts.")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--leaderboard-output", default="outputs/metrics/leaderboard.csv")
    parser.add_argument("--combined-output", default="outputs/predictions/all_predictions.parquet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--align-intersection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_compare")
    leaderboard_path = export_leaderboard(
        args.predictions_dir,
        args.leaderboard_output,
        split=args.split or None,
        align_intersection=args.align_intersection,
    )
    combined_path = export_all_predictions(args.predictions_dir, args.combined_output)
    logger.info("Comparison artifacts written: leaderboard=%s combined=%s", leaderboard_path, combined_path)


if __name__ == "__main__":
    main()
