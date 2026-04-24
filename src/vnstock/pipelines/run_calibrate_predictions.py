from __future__ import annotations

import argparse
from pathlib import Path

from vnstock.data.loaders import load_prediction_files
from vnstock.models.common.calibration import calibrate_direction_threshold
from vnstock.utils.io import load_table, save_table, write_json
from vnstock.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalibrate direction thresholds in prediction files.")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--min-improvement", type=float, default=0.001)
    parser.add_argument("--min-positive-rate", type=float, default=0.2)
    parser.add_argument("--max-positive-rate", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_calibrate_predictions")
    for path in load_prediction_files(args.predictions_dir):
        frame = load_table(path)
        if args.valid_split not in set(frame["split"].astype(str)):
            continue
        score_column = "direction_score" if "direction_score" in frame.columns else "y_pred"
        default_threshold = 0.5 if score_column == "direction_score" else 0.0
        valid = frame.loc[frame["split"].astype(str) == args.valid_split].copy()
        calibration = calibrate_direction_threshold(
            valid,
            score_column=score_column,
            y_true_column="y_true",
            default_threshold=default_threshold,
            min_improvement=args.min_improvement,
            min_positive_rate=args.min_positive_rate,
            max_positive_rate=args.max_positive_rate,
        )
        frame["direction_score_column"] = calibration.score_column
        frame["direction_threshold"] = calibration.threshold
        save_table(frame, path)
        calibration_path = Path(args.predictions_dir) / f"{path.stem.replace('_predictions', '')}_calibration.json"
        write_json(
            {
                "score_column": calibration.score_column,
                "threshold": calibration.threshold,
                "valid_accuracy": calibration.valid_accuracy,
                "default_threshold": calibration.default_threshold,
                "default_valid_accuracy": calibration.default_valid_accuracy,
                "predicted_positive_rate": calibration.predicted_positive_rate,
            },
            calibration_path,
        )
        logger.info("%s threshold=%s score=%s", path.name, calibration.threshold, score_column)


if __name__ == "__main__":
    main()
