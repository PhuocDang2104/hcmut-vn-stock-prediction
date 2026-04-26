from __future__ import annotations

import pandas as pd
import torch

from vnstock.models.common.base_trainer import BaseTrainer, TrainingArtifacts
from vnstock.models.common.calibration import calibrate_direction_threshold
from vnstock.models.common.sequence_data import (
    build_effective_feature_columns,
    build_scaled_sequence_splits,
    cap_sequence_samples,
    load_shared_feature_panel,
    load_shared_metadata,
    resolve_feature_columns,
)
from vnstock.models.common.torch_training import (
    fit_regression_model,
    predict_multitask_model,
    resolve_device,
)
from vnstock.models.xlstm_ts.inference import build_prediction_frame
from vnstock.models.xlstm_ts.model import build_model, build_model_spec
from vnstock.utils.io import ensure_dir, save_table, write_json
from vnstock.utils.logging import get_logger
from vnstock.utils.paths import path_for
from vnstock.utils.schema import PredictionContext
from vnstock.utils.seed import set_global_seed


class XLSTMTrainer(BaseTrainer):
    def __init__(self, config: dict) -> None:
        super().__init__(model_name="xlstm_ts", config=config)
        self.logger = get_logger("vnstock.models.xlstm_ts.trainer")

    def fit(self) -> TrainingArtifacts:
        seed = int(self.config.get("seed", 42))
        set_global_seed(seed)
        run_id, checkpoint_dir = self.initialize_run()

        shared_meta = load_shared_metadata()
        feature_panel = load_shared_feature_panel()
        symbol_filter = [str(symbol) for symbol in self.config.get("symbol_filter", [])]
        if symbol_filter:
            feature_panel = feature_panel.loc[feature_panel["symbol"].astype(str).isin(symbol_filter)].copy()
            if feature_panel.empty:
                raise ValueError(f"No rows found for symbol_filter={symbol_filter}.")
        feature_columns = resolve_feature_columns(self.config, shared_meta, key="feature_columns")
        wavelet_config = self.config.get("wavelet_denoise")
        target_scaling = self.config.get("target_scaling")
        effective_feature_columns = build_effective_feature_columns(feature_columns, wavelet_config)
        target_column = str(self.config["target"])
        context_length = int(self.config["context_length"])

        sequence_splits, scalers = build_scaled_sequence_splits(
            feature_panel=feature_panel,
            feature_columns=feature_columns,
            target_column=target_column,
            lookback=context_length,
            wavelet_config=wavelet_config,
            target_scaling=target_scaling,
        )
        sequence_splits["train"] = cap_sequence_samples(
            sequence_splits["train"],
            max_samples=self.config.get("train_sample_cap"),
            seed=seed,
        )
        sequence_splits["valid"] = cap_sequence_samples(
            sequence_splits["valid"],
            max_samples=self.config.get("valid_sample_cap"),
            seed=seed + 1,
        )
        sequence_splits["test"] = cap_sequence_samples(
            sequence_splits["test"],
            max_samples=self.config.get("test_sample_cap"),
            seed=seed + 2,
        )

        model = build_model(self.config, num_features=len(effective_feature_columns))
        device = resolve_device(self.config.get("device", "auto"))
        training_result = fit_regression_model(
            model,
            sequence_splits["train"],
            sequence_splits["valid"],
            batch_size=int(self.config["batch_size"]),
            eval_batch_size=int(self.config.get("eval_batch_size", self.config["batch_size"])),
            epochs=int(self.config["epochs"]),
            learning_rate=float(self.config["learning_rate"]),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
            patience=int(self.config.get("patience", 3)),
            huber_delta=float(self.config.get("huber_delta", 0.05)),
            direction_loss_weight=float(self.config.get("direction_loss_weight", 0.0)),
            direction_pos_weight=self.config.get("direction_pos_weight"),
            direction_large_move_quantile=self.config.get("direction_large_move_quantile"),
            direction_large_move_weight=float(self.config.get("direction_large_move_weight", 1.0)),
            direction_abs_return_weight=float(self.config.get("direction_abs_return_weight", 0.0)),
            rank_loss_weight=float(self.config.get("rank_loss_weight", 0.0)),
            rank_loss_min_target_diff=float(self.config.get("rank_loss_min_target_diff", 0.0)),
            checkpoint_metric=str(self.config.get("checkpoint_metric", "valid_loss")),
            checkpoint_mode=self.config.get("checkpoint_mode"),
            device=device,
            input_dtype=torch.float32,
            logger=self.logger,
        )

        prediction_context = PredictionContext(
            model_family=self.model_name,
            model_version=str(self.config.get("model_version", "residual_lstm_multitask_v1")),
            target_name=target_column,
            horizon=int(self.config.get("horizon", shared_meta.get("horizon", 5))),
            run_id=run_id,
        )
        raw_prediction_frames: dict[str, pd.DataFrame] = {}
        for split_name in ("train", "valid", "test"):
            sample_set = sequence_splits[split_name]
            prediction_result = predict_multitask_model(
                model,
                sample_set,
                batch_size=int(self.config.get("eval_batch_size", self.config["batch_size"])),
                device=device,
                input_dtype=torch.float32,
            )
            frame = sample_set.meta.copy()
            target_values, prediction_values = _restore_target_scale(sample_set, prediction_result.y_pred)
            frame[target_column] = target_values
            frame["y_pred"] = prediction_values
            if prediction_result.direction_score is not None:
                frame["direction_score"] = prediction_result.direction_score
            raw_prediction_frames[split_name] = frame

        calibrations = [
            calibrate_direction_threshold(
                raw_prediction_frames["valid"],
                score_column=score_column,
                y_true_column=target_column,
                default_threshold=0.5 if score_column == "direction_score" else 0.0,
                min_improvement=float(self.config.get("direction_calibration_min_improvement", 0.0)),
                min_positive_rate=float(self.config.get("direction_calibration_min_positive_rate", 0.2)),
                max_positive_rate=float(self.config.get("direction_calibration_max_positive_rate", 0.8)),
                optimize_metric=str(self.config.get("direction_calibration_metric", "accuracy")),
            )
            for score_column in _direction_score_candidates(
                raw_prediction_frames["valid"],
                source=str(self.config.get("direction_score_source", "direction_score")),
            )
        ]
        calibration = max(calibrations, key=lambda item: item.optimized_metric_value)

        prediction_frames: list[pd.DataFrame] = []
        for split_name in ("train", "valid", "test"):
            frame = raw_prediction_frames[split_name]
            prediction_frame = build_prediction_frame(frame, prediction_context)
            if calibration.score_column in frame.columns:
                prediction_frame[calibration.score_column] = frame[calibration.score_column].to_numpy(dtype=float)
            prediction_frame["direction_score_column"] = calibration.score_column
            prediction_frame["direction_threshold"] = calibration.threshold
            prediction_frames.append(prediction_frame)

        combined_predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions_root = ensure_dir(self.config.get("predictions_dir", path_for("outputs_root") / "predictions"))
        predictions_path = save_table(
            combined_predictions,
            predictions_root / f"{self.model_name}_predictions.parquet",
        )
        calibration_path = write_json(
            {
                "score_column": calibration.score_column,
                "threshold": calibration.threshold,
                "valid_accuracy": calibration.valid_accuracy,
                "valid_balanced_accuracy": calibration.valid_balanced_accuracy,
                "default_threshold": calibration.default_threshold,
                "default_valid_accuracy": calibration.default_valid_accuracy,
                "default_valid_balanced_accuracy": calibration.default_valid_balanced_accuracy,
                "predicted_positive_rate": calibration.predicted_positive_rate,
                "optimized_metric": calibration.optimized_metric,
                "optimized_metric_value": calibration.optimized_metric_value,
            },
            predictions_root / f"{self.model_name}_calibration.json",
        )

        checkpoint_path = checkpoint_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config,
                "feature_columns": feature_columns,
                "effective_feature_columns": effective_feature_columns,
                "context_length": context_length,
                "direction_loss_weight": float(self.config.get("direction_loss_weight", 0.0)),
                "rank_loss_weight": float(self.config.get("rank_loss_weight", 0.0)),
                "rank_loss_min_target_diff": float(self.config.get("rank_loss_min_target_diff", 0.0)),
            },
            checkpoint_path,
        )
        write_json(
            {
                "feature_columns": feature_columns,
                "effective_feature_columns": effective_feature_columns,
                "scalers": scalers,
                "context_length": context_length,
                "wavelet_denoise": wavelet_config,
                "target_scaling": target_scaling,
            },
            checkpoint_dir / "preprocess.json",
        )
        history_path = write_json(
            {
                "history": training_result.history,
                "best_valid_loss": training_result.best_valid_loss,
            },
            checkpoint_dir / "training_history.json",
        )
        spec_path = write_json(
            build_model_spec(self.config) | {"num_features": len(effective_feature_columns)},
            checkpoint_dir / "model_spec.json",
        )

        details = {
            "device": str(device),
            "feature_columns": feature_columns,
            "effective_feature_columns": effective_feature_columns,
            "context_length": context_length,
            "num_features": len(effective_feature_columns),
            "symbol_filter": symbol_filter,
            "best_valid_loss": training_result.best_valid_loss,
            "direction_calibration": {
                "score_column": calibration.score_column,
                "threshold": calibration.threshold,
                "valid_accuracy": calibration.valid_accuracy,
                "valid_balanced_accuracy": calibration.valid_balanced_accuracy,
                "default_threshold": calibration.default_threshold,
                "default_valid_accuracy": calibration.default_valid_accuracy,
                "default_valid_balanced_accuracy": calibration.default_valid_balanced_accuracy,
                "optimized_metric": calibration.optimized_metric,
                "optimized_metric_value": calibration.optimized_metric_value,
            },
            "checkpoint_path": str(checkpoint_path),
            "predictions_path": str(predictions_path),
            "calibration_path": str(calibration_path),
            "history_path": str(history_path),
            "model_spec_path": str(spec_path),
            "sample_sizes": {
                split_name: sequence_splits[split_name].size for split_name in sequence_splits
            },
        }
        return self.finalize_run(
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            status="trained",
            note="xLSTM-TS benchmark run completed with residual recurrent multitask backbone.",
            details=details,
        )


def _direction_score_candidates(frame: pd.DataFrame, *, source: str) -> list[str]:
    available = ["y_pred"]
    if "direction_score" in frame.columns:
        available.insert(0, "direction_score")
    if source == "auto":
        return available
    if source not in available:
        raise ValueError(f"direction_score_source={source!r} is unavailable. Available: {available}")
    return [source]


def _restore_target_scale(sample_set, predictions):
    if sample_set.meta.empty or not {"target_raw", "target_mean", "target_std"}.issubset(sample_set.meta.columns):
        return sample_set.y, predictions
    means = sample_set.meta["target_mean"].to_numpy(dtype=float)
    stds = sample_set.meta["target_std"].to_numpy(dtype=float)
    target_values = sample_set.meta["target_raw"].to_numpy(dtype=float)
    prediction_values = predictions * stds + means
    return target_values, prediction_values
