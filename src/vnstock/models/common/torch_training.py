from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    device: torch.device,
    input_dtype: torch.dtype,
    logger: Any | None = None,
) -> TrainingResult:
    train_loader = _make_loader(train_set, batch_size=batch_size, shuffle=True, input_dtype=input_dtype)
    valid_loader = _make_loader(valid_set, batch_size=eval_batch_size, shuffle=False, input_dtype=input_dtype)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    regression_criterion = nn.HuberLoss(delta=huber_delta)
    direction_criterion = nn.BCEWithLogitsLoss()

    best_valid_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_regression_losses: list[float] = []
        train_direction_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(batch_x)
            preds, direction_logits = _unpack_model_output(model_output)
            regression_loss = regression_criterion(preds, batch_y)
            direction_loss = torch.zeros((), device=device)
            if direction_logits is not None and direction_loss_weight > 0:
                direction_target = (batch_y > 0).to(torch.float32)
                direction_loss = direction_criterion(direction_logits, direction_target)
            loss = regression_loss + direction_loss_weight * direction_loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_regression_losses.append(float(regression_loss.detach().cpu()))
            train_direction_losses.append(float(direction_loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        valid_metrics = evaluate_loss(
            model,
            valid_loader,
            regression_criterion=regression_criterion,
            direction_criterion=direction_criterion,
            direction_loss_weight=direction_loss_weight,
            device=device,
        )
        valid_loss = valid_metrics["total_loss"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_regression_loss": float(np.mean(train_regression_losses)),
                "train_direction_loss": float(np.mean(train_direction_losses)),
                "valid_loss": valid_loss,
                "valid_regression_loss": valid_metrics["regression_loss"],
                "valid_direction_loss": valid_metrics["direction_loss"],
                "valid_direction_accuracy": valid_metrics["direction_accuracy"],
            }
        )
        if logger is not None:
            logger.info(
                "%s epoch %s/%s train_loss=%.6f valid_loss=%.6f valid_dir_acc=%.4f",
                model.__class__.__name__,
                epoch,
                epochs,
                train_loss,
                valid_loss,
                valid_metrics["direction_accuracy"],
            )

        if valid_loss < best_valid_loss:
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
    device: torch.device,
) -> dict[str, float]:
    losses: list[float] = []
    regression_losses: list[float] = []
    direction_losses: list[float] = []
    direction_hits: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_y = batch_y.to(device)
            preds, direction_logits = _unpack_model_output(model(batch_x.to(device)))
            regression_loss = regression_criterion(preds, batch_y)
            direction_loss = torch.zeros((), device=device)
            if direction_logits is not None:
                direction_target = (batch_y > 0).to(torch.float32)
                direction_loss = direction_criterion(direction_logits, direction_target)
                direction_pred = torch.sigmoid(direction_logits) > 0.5
                direction_hits.append(float((direction_pred == direction_target.bool()).float().mean()))
            loss = regression_loss + direction_loss_weight * direction_loss
            losses.append(float(loss.detach().cpu()))
            regression_losses.append(float(regression_loss.detach().cpu()))
            direction_losses.append(float(direction_loss.detach().cpu()))
    return {
        "total_loss": float(np.mean(losses)) if losses else float("nan"),
        "regression_loss": float(np.mean(regression_losses)) if regression_losses else float("nan"),
        "direction_loss": float(np.mean(direction_losses)) if direction_losses else float("nan"),
        "direction_accuracy": float(np.mean(direction_hits)) if direction_hits else float("nan"),
    }


def _unpack_model_output(model_output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_output, dict):
        return model_output["prediction"], model_output.get("direction_logit")
    return model_output, None


def _make_loader(
    sample_set: SequenceSampleSet,
    *,
    batch_size: int,
    shuffle: bool,
    input_dtype: torch.dtype,
) -> DataLoader:
    x_tensor = torch.tensor(sample_set.X, dtype=input_dtype)
    y_tensor = torch.tensor(sample_set.y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
