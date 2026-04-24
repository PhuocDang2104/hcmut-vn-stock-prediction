from __future__ import annotations

import argparse

from vnstock.data.dataset_builder import build_all_datasets
from vnstock.data.ingest import ingest_repo_csvs
from vnstock.evaluation.compare import export_all_predictions
from vnstock.evaluation.leaderboard import export_leaderboard
from vnstock.models.itransformer.trainer import ITransformerTrainer
from vnstock.models.kronos.trainer import KronosTrainer
from vnstock.models.xlstm_ts.trainer import XLSTMTrainer
from vnstock.utils.io import load_table, read_yaml
from vnstock.utils.logging import get_logger
from vnstock.utils.paths import path_for
from vnstock.utils.seed import set_global_seed


TRAINER_CONFIGS = {
    "xlstm_ts": ("configs/models/xlstm_ts.yaml", XLSTMTrainer),
    "itransformer": ("configs/models/itransformer.yaml", ITransformerTrainer),
    "kronos": ("configs/models/kronos.yaml", KronosTrainer),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train xLSTM-TS and iTransformer, then export a test leaderboard."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xlstm_ts", "itransformer"],
        choices=sorted(TRAINER_CONFIGS),
    )
    parser.add_argument("--dataset-config", default="configs/data/dataset_daily.yaml")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--use-interim", action="store_true")
    parser.add_argument("--leaderboard-output", default="outputs/metrics/leaderboard.csv")
    parser.add_argument("--combined-output", default="outputs/predictions/all_predictions.parquet")
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("vnstock.pipelines.run_benchmark")
    set_global_seed(args.seed)

    if args.rebuild_dataset:
        dataset_config = read_yaml(args.dataset_config)
        if args.use_interim and (path_for("interim_root") / "cleaned_panel.parquet").exists():
            raw_panel = load_table(path_for("interim_root") / "cleaned_panel.parquet")
        else:
            ingest_repo_csvs(dataset_config)
            raw_panel = load_table(path_for("interim_root") / "cleaned_panel.parquet")
        exports = build_all_datasets(raw_panel, dataset_config)
        logger.info("Dataset build complete: %s", exports)

    artifacts = {}
    for model_name in args.models:
        config_path, trainer_cls = TRAINER_CONFIGS[model_name]
        trainer = trainer_cls(read_yaml(config_path))
        artifact = trainer.fit()
        artifacts[model_name] = artifact.manifest_path
        logger.info("%s finished: manifest=%s", model_name, artifact.manifest_path)

    leaderboard_path = export_leaderboard(
        args.predictions_dir,
        args.leaderboard_output,
        split=args.split or None,
    )
    combined_path = export_all_predictions(args.predictions_dir, args.combined_output)
    logger.info(
        "Benchmark suite complete: leaderboard=%s combined=%s manifests=%s",
        leaderboard_path,
        combined_path,
        artifacts,
    )


if __name__ == "__main__":
    main()
