from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vnstock.models.common.base_predictor import standardize_prediction_output
from vnstock.models.common.sequence_data import load_shared_feature_panel
from vnstock.utils.io import ensure_dir, load_table, save_table
from vnstock.utils.paths import path_for
from vnstock.utils.registry import generate_run_id
from vnstock.utils.schema import PredictionContext


@dataclass(frozen=True)
class KronosZeroShotConfig:
    model_name: str
    tokenizer_name: str
    context_length: int
    horizon: int
    split: str
    device: str
    sample_count: int
    temperature: float
    top_p: float
    max_symbols: int | None
    latest_only: bool
    kronos_repo: str | None


def run_kronos_zero_shot(config: dict[str, Any]) -> Path:
    zero_shot_config = _parse_config(config)
    Kronos, KronosTokenizer, KronosPredictor = _load_kronos_classes(zero_shot_config.kronos_repo)

    tokenizer = KronosTokenizer.from_pretrained(zero_shot_config.tokenizer_name)
    model = Kronos.from_pretrained(zero_shot_config.model_name)
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=zero_shot_config.device,
        max_context=zero_shot_config.context_length,
    )

    feature_panel = load_shared_feature_panel()
    raw_panel = load_table(path_for("interim_root") / "cleaned_panel.parquet")
    records = _predict_records(
        predictor=predictor,
        feature_panel=feature_panel,
        raw_panel=raw_panel,
        config=zero_shot_config,
    )
    prediction_frame = pd.DataFrame(records)
    if prediction_frame.empty:
        raise ValueError("Kronos zero-shot did not produce any prediction rows.")

    run_id = generate_run_id("kronos")
    context = PredictionContext(
        model_family="kronos",
        model_version=zero_shot_config.model_name,
        target_name="target_ret_5d",
        horizon=zero_shot_config.horizon,
        run_id=run_id,
    )
    output = standardize_prediction_output(prediction_frame, context)
    output["direction_score_column"] = "y_pred"
    output["direction_threshold"] = 0.0
    predictions_root = ensure_dir(path_for("outputs_root") / "predictions")
    return save_table(output, predictions_root / "kronos_predictions.parquet")


def _parse_config(config: dict[str, Any]) -> KronosZeroShotConfig:
    return KronosZeroShotConfig(
        model_name=str(config.get("pretrained_model", "NeoQuasar/Kronos-small")),
        tokenizer_name=str(config.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base")),
        context_length=int(config.get("context_length", 64)),
        horizon=int(config.get("prediction_length", config.get("horizon", 5))),
        split=str(config.get("split", "test")),
        device=str(config.get("device", "cpu")),
        sample_count=int(config.get("sample_count", 1)),
        temperature=float(config.get("temperature", 1.0)),
        top_p=float(config.get("top_p", 0.9)),
        max_symbols=config.get("max_symbols"),
        latest_only=bool(config.get("latest_only", True)),
        kronos_repo=config.get("kronos_repo"),
    )


def _load_kronos_classes(kronos_repo: str | None):
    if kronos_repo:
        repo_path = str(Path(kronos_repo).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        module = importlib.import_module("model")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Official Kronos module is not importable. Clone https://github.com/shiyu-coder/Kronos "
            "and pass --kronos-repo <path>, or install it in the active Python environment."
        ) from exc
    return module.Kronos, module.KronosTokenizer, module.KronosPredictor


def _predict_records(
    *,
    predictor: Any,
    feature_panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    config: KronosZeroShotConfig,
) -> list[dict[str, Any]]:
    feature_panel = feature_panel.loc[feature_panel["split"].astype(str) == config.split].copy()
    raw_panel = raw_panel.sort_values(["symbol", "date"]).copy()
    raw_panel["date"] = pd.to_datetime(raw_panel["date"])
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])

    symbols = sorted(feature_panel["symbol"].astype(str).unique())
    if config.max_symbols is not None:
        symbols = symbols[: int(config.max_symbols)]

    records: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_features = feature_panel.loc[feature_panel["symbol"].astype(str) == symbol]
        if config.latest_only:
            symbol_features = symbol_features.tail(1)
        symbol_raw = raw_panel.loc[raw_panel["symbol"].astype(str) == symbol].reset_index(drop=True)
        raw_date_to_index = {date: idx for idx, date in enumerate(symbol_raw["date"])}

        for _, row in symbol_features.iterrows():
            raw_idx = raw_date_to_index.get(row["date"])
            if raw_idx is None or raw_idx < config.context_length - 1:
                continue
            if raw_idx + config.horizon >= len(symbol_raw):
                continue
            context = symbol_raw.iloc[raw_idx - config.context_length + 1 : raw_idx + 1]
            future = symbol_raw.iloc[raw_idx + 1 : raw_idx + config.horizon + 1]
            x_df = context[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=context["date"].reset_index(drop=True),
                y_timestamp=future["date"].reset_index(drop=True),
                pred_len=config.horizon,
                T=config.temperature,
                top_p=config.top_p,
                sample_count=config.sample_count,
                verbose=False,
            )
            predicted_close = float(pred_df["close"].iloc[-1])
            current_close = float(row["close"])
            records.append(
                {
                    "symbol": symbol,
                    "date": row["date"],
                    "split": config.split,
                    "target_ret_5d": float(row["target_ret_5d"]),
                    "y_pred": predicted_close / current_close - 1.0,
                }
            )
    return records
