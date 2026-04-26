from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vnstock.models.common.sequence_data import SequenceSampleSet


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_valid_loss: float


@dataclass
class PredictionResult:
    y_pred: np.ndarray
    direction_score: np.ndarray | None = None


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested.lower() != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_regression_model(
    model: nn.Module,
    train_set: SequenceSampleSet,
    valid_set: SequenceSampleSet,
    *,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    huber_delta: float,
    direction_loss_weight: float = 0.0,
    direction_pos_weight: float | str | None = None,
    direction_large_move_quantile: float | None = None,
    direction_large_move_weight: float = 1.0,
    direction_abs_return_weight: float = 0.0,
    rank_loss_weight: float = 0.0,
    rank_loss_min_target_diff: float = 0.0,
    checkpoint_metric: str = "valid_loss",
    checkpoint_mode: str | None = None,
    device: torch.device,
    input_dtype: torch.dtype,
    logger: Any | None = None,
) -> TrainingResult:
    rank_enabled = rank_loss_weight > 0
    train_loader = _make_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        input_dtype=input_dtype,
        include_date_ids=rank_enabled,
        group_by_date=rank_enabled,
    )
    valid_loader = _make_loader(
        valid_set,
        batch_size=eval_batch_size,
        shuffle=False,
        input_dtype=input_dtype,
        include_date_ids=rank_enabled,
        group_by_date=rank_enabled,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    regression_criterion = nn.HuberLoss(delta=huber_delta)
    direction_criterion = nn.BCEWithLogitsLoss(
        pos_weight=_direction_pos_weight_tensor(train_set.y, direction_pos_weight, device=device),
        reduction="none",
    )
    large_move_threshold = _large_move_threshold(train_set.y, direction_large_move_quantile)
    monitor_mode = checkpoint_mode or ("min" if checkpoint_metric.endswith("loss") else "max")

    best_valid_loss = float("inf")
    best_monitor_value = float("inf") if monitor_mode == "min" else -float("inf")
    best_state = deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_regression_losses: list[float] = []
        train_direction_losses: list[float] = []
        train_rank_losses: list[float] = []
        for batch in train_loader:
            batch_x, batch_y, batch_date_ids = _unpack_batch(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if batch_date_ids is not None:
                batch_date_ids = batch_date_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(batch_x)
            preds, direction_logits = _unpack_model_output(model_output)
            regression_loss = regression_criterion(preds, batch_y)
            direction_loss = torch.zeros((), device=device)
            if direction_logits is not None and direction_loss_weight > 0:
                direction_target = (batch_y > 0).to(torch.float32)
                direction_loss = weighted_direction_loss(
                    direction_logits,
                    direction_target,
                    batch_y,
                    criterion=direction_criterion,
                    large_move_threshold=large_move_threshold,
                    large_move_weight=direction_large_move_weight,
                    abs_return_weight=direction_abs_return_weight,
                )
            rank_loss = torch.zeros((), device=device)
            if batch_date_ids is not None and rank_loss_weight > 0:
                rank_loss = pairwise_rank_loss(
                    preds,
                    batch_y,
                    batch_date_ids,
                    min_target_diff=rank_loss_min_target_diff,
                )
            loss = regression_loss + direction_loss_weight * direction_loss + rank_loss_weight * rank_loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_regression_losses.append(float(regression_loss.detach().cpu()))
            train_direction_losses.append(float(direction_loss.detach().cpu()))
            train_rank_losses.append(float(rank_loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        valid_metrics = evaluate_loss(
            model,
            valid_loader,
            regression_criterion=regression_criterion,
            direction_criterion=direction_criterion,
            direction_loss_weight=direction_loss_weight,
            large_move_threshold=large_move_threshold,
            direction_large_move_weight=direction_large_move_weight,
            direction_abs_return_weight=direction_abs_return_weight,
            rank_loss_weight=rank_loss_weight,
            rank_loss_min_target_diff=rank_loss_min_target_diff,
            device=device,
        )
        valid_loss = valid_metrics["total_loss"]
        history_row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_regression_loss": float(np.mean(train_regression_losses)),
            "train_direction_loss": float(np.mean(train_direction_losses)),
            "train_rank_loss": float(np.mean(train_rank_losses)),
            "valid_loss": valid_loss,
            "valid_regression_loss": valid_metrics["regression_loss"],
            "valid_direction_loss": valid_metrics["direction_loss"],
            "valid_rank_loss": valid_metrics["rank_loss"],
            "valid_direction_accuracy": valid_metrics["direction_accuracy"],
            "valid_balanced_direction_accuracy": valid_metrics["balanced_direction_accuracy"],
        }
        history.append(history_row)
        if logger is not None:
            logger.info(
                "%s epoch %s/%s train_loss=%.6f valid_loss=%.6f valid_dir_acc=%.4f valid_bal_acc=%.4f",
                model.__class__.__name__,
                epoch,
                epochs,
                train_loss,
                valid_loss,
                valid_metrics["direction_accuracy"],
                valid_metrics["balanced_direction_accuracy"],
            )

        monitor_value = float(history_row.get(checkpoint_metric, valid_loss))
        if _is_better(monitor_value, best_monitor_value, mode=monitor_mode):
            best_monitor_value = monitor_value
            best_valid_loss = valid_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(history=history, best_valid_loss=best_valid_loss)


def predict_regression_model(
    model: nn.Module,
    sample_set: SequenceSampleSet,
    *,
    batch_size: int,
    device: torch.device,
    input_dtype: torch.dtype,
) -> np.ndarray:
    return predict_multitask_model(
        model,
        sample_set,
        batch_size=batch_size,
        device=device,
        input_dtype=input_dtype,
    ).y_pred


def predict_multitask_model(
    model: nn.Module,
    sample_set: SequenceSampleSet,
    *,
    batch_size: int,
    device: torch.device,
    input_dtype: torch.dtype,
) -> PredictionResult:
    loader = _make_loader(sample_set, batch_size=batch_size, shuffle=False, input_dtype=input_dtype)
    prediction_outputs: list[np.ndarray] = []
    direction_outputs: list[np.ndarray] = []
    has_direction_head = False
    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            preds, direction_logits = _unpack_model_output(model(batch_x.to(device)))
            prediction_outputs.append(preds.detach().cpu().numpy().astype(np.float32, copy=False))
            if direction_logits is not None:
                has_direction_head = True
                direction_score = torch.sigmoid(direction_logits).detach().cpu().numpy()
                direction_outputs.append(direction_score.astype(np.float32, copy=False))
    if not prediction_outputs:
        return PredictionResult(y_pred=np.empty((0,), dtype=np.float32))
    y_pred = np.concatenate(prediction_outputs, axis=0)
    if not has_direction_head:
        return PredictionResult(y_pred=y_pred)
    return PredictionResult(
        y_pred=y_pred,
        direction_score=np.concatenate(direction_outputs, axis=0),
    )


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    regression_criterion: nn.Module,
    direction_criterion: nn.Module,
    direction_loss_weight: float,
    large_move_threshold: float | None = None,
    direction_large_move_weight: float = 1.0,
    direction_abs_return_weight: float = 0.0,
    rank_loss_weight: float = 0.0,
    rank_loss_min_target_diff: float = 0.0,
    device: torch.device,
) -> dict[str, float]:
    losses: list[float] = []
    regression_losses: list[float] = []
    direction_losses: list[float] = []
    rank_losses: list[float] = []
    direction_hits: list[float] = []
    direction_targets: list[np.ndarray] = []
    direction_predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y, batch_date_ids = _unpack_batch(batch)
            batch_y = batch_y.to(device)
            if batch_date_ids is not None:
                batch_date_ids = batch_date_ids.to(device)
            preds, direction_logits = _unpack_model_output(model(batch_x.to(device)))
            regression_loss = regression_criterion(preds, batch_y)
            direction_loss = torch.zeros((), device=device)
            if direction_logits is not None:
                direction_target = (batch_y > 0).to(torch.float32)
                direction_loss = weighted_direction_loss(
                    direction_logits,
                    direction_target,
                    batch_y,
                    criterion=direction_criterion,
                    large_move_threshold=large_move_threshold,
                    large_move_weight=direction_large_move_weight,
                    abs_return_weight=direction_abs_return_weight,
                )
                direction_pred = torch.sigmoid(direction_logits) > 0.5
                direction_hits.append(float((direction_pred == direction_target.bool()).float().mean()))
                direction_targets.append(direction_target.detach().cpu().numpy().astype(bool, copy=False))
                direction_predictions.append(direction_pred.detach().cpu().numpy().astype(bool, copy=False))
            rank_loss = torch.zeros((), device=device)
            if batch_date_ids is not None and rank_loss_weight > 0:
                rank_loss = pairwise_rank_loss(
                    preds,
                    batch_y,
                    batch_date_ids,
                    min_target_diff=rank_loss_min_target_diff,
                )
            loss = regression_loss + direction_loss_weight * direction_loss + rank_loss_weight * rank_loss
            losses.append(float(loss.detach().cpu()))
            regression_losses.append(float(regression_loss.detach().cpu()))
            direction_losses.append(float(direction_loss.detach().cpu()))
            rank_losses.append(float(rank_loss.detach().cpu()))
    return {
        "total_loss": float(np.mean(losses)) if losses else float("nan"),
        "regression_loss": float(np.mean(regression_losses)) if regression_losses else float("nan"),
        "direction_loss": float(np.mean(direction_losses)) if direction_losses else float("nan"),
        "rank_loss": float(np.mean(rank_losses)) if rank_losses else float("nan"),
        "direction_accuracy": float(np.mean(direction_hits)) if direction_hits else float("nan"),
        "balanced_direction_accuracy": _balanced_accuracy(direction_targets, direction_predictions),
    }


def weighted_direction_loss(
    logits: torch.Tensor,
    direction_target: torch.Tensor,
    returns: torch.Tensor,
    *,
    criterion: nn.Module,
    large_move_threshold: float | None = None,
    large_move_weight: float = 1.0,
    abs_return_weight: float = 0.0,
) -> torch.Tensor:
    losses = criterion(logits, direction_target)
    weights = torch.ones_like(losses)
    if large_move_threshold is not None and large_move_weight != 1.0:
        weights = torch.where(torch.abs(returns) >= large_move_threshold, large_move_weight, weights)
    if abs_return_weight > 0:
        weights = weights * (1.0 + abs_return_weight * torch.abs(returns))
    return torch.sum(losses * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def pairwise_rank_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    date_ids: torch.Tensor,
    *,
    min_target_diff: float = 0.0,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for date_id in torch.unique(date_ids):
        mask = date_ids == date_id
        if int(mask.sum()) < 2:
            continue
        pred_values = preds[mask]
        target_values = targets[mask]
        target_diff = target_values[:, None] - target_values[None, :]
        pair_mask = target_diff > min_target_diff
        if not bool(pair_mask.any()):
            continue
        pred_diff = pred_values[:, None] - pred_values[None, :]
        losses.append(torch.nn.functional.softplus(-pred_diff[pair_mask]).mean())
    if not losses:
        return torch.zeros((), device=preds.device)
    return torch.stack(losses).mean()


def _direction_pos_weight_tensor(
    targets: np.ndarray,
    direction_pos_weight: float | str | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if direction_pos_weight is None:
        return None
    if isinstance(direction_pos_weight, str):
        if direction_pos_weight != "auto":
            raise ValueError(f"Unsupported direction_pos_weight: {direction_pos_weight}")
        positives = float(np.sum(targets > 0))
        negatives = float(np.sum(targets <= 0))
        if positives <= 0 or negatives <= 0:
            return None
        return torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    if direction_pos_weight <= 0:
        return None
    return torch.tensor(float(direction_pos_weight), dtype=torch.float32, device=device)


def _large_move_threshold(targets: np.ndarray, quantile: float | None) -> float | None:
    if quantile is None:
        return None
    if not 0.0 < quantile < 1.0:
        raise ValueError("direction_large_move_quantile must be between 0 and 1.")
    values = np.abs(np.asarray(targets, dtype=np.float32))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(np.quantile(values, quantile))


def _balanced_accuracy(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
) -> float:
    if not targets or not predictions:
        return float("nan")
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    positive_mask = y_true
    negative_mask = ~y_true
    if not positive_mask.any() or not negative_mask.any():
        return float("nan")
    true_positive_rate = float(np.mean(y_pred[positive_mask]))
    true_negative_rate = float(np.mean(~y_pred[negative_mask]))
    return 0.5 * (true_positive_rate + true_negative_rate)


def _is_better(value: float, best_value: float, *, mode: str) -> bool:
    if not np.isfinite(value):
        return False
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"Unsupported checkpoint mode: {mode}")


def _unpack_model_output(model_output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_output, dict):
        return model_output["prediction"], model_output.get("direction_logit")
    return model_output, None


def _unpack_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _make_loader(
    sample_set: SequenceSampleSet,
    *,
    batch_size: int,
    shuffle: bool,
    input_dtype: torch.dtype,
    include_date_ids: bool = False,
    group_by_date: bool = False,
) -> DataLoader:
    x_tensor = torch.tensor(sample_set.X, dtype=input_dtype)
    y_tensor = torch.tensor(sample_set.y, dtype=torch.float32)
    if include_date_ids:
        date_ids = torch.tensor(_date_codes(sample_set), dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor, date_ids)
    else:
        dataset = TensorDataset(x_tensor, y_tensor)
    if group_by_date and include_date_ids:
        batches = _date_grouped_batches(sample_set, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batches, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _date_codes(sample_set: SequenceSampleSet) -> np.ndarray:
    if sample_set.meta.empty or "date" not in sample_set.meta.columns:
        return np.zeros(sample_set.size, dtype=np.int64)
    codes, _ = pd.factorize(pd.to_datetime(sample_set.meta["date"]), sort=True)
    return codes.astype(np.int64, copy=False)


def _date_grouped_batches(
    sample_set: SequenceSampleSet,
    *,
    batch_size: int,
    shuffle: bool,
) -> list[list[int]]:
    date_codes = _date_codes(sample_set)
    groups = [np.flatnonzero(date_codes == date_code).tolist() for date_code in np.unique(date_codes)]
    if shuffle:
        rng = np.random.default_rng(42)
        rng.shuffle(groups)

    batches: list[list[int]] = []
    current: list[int] = []
    for group in groups:
        if len(group) >= batch_size:
            if current:
                batches.append(current)
                current = []
            for start in range(0, len(group), batch_size):
                batches.append(group[start : start + batch_size])
            continue
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches
