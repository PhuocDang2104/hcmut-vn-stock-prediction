from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
KRONOS_REPO = ROOT / "external" / "Kronos"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(KRONOS_REPO) not in sys.path:
    sys.path.insert(0, str(KRONOS_REPO))

from vnstock.models.common.base_predictor import standardize_prediction_output  # noqa: E402
from vnstock.models.common.sequence_data import load_shared_feature_panel  # noqa: E402
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402
from vnstock.utils.paths import path_for  # noqa: E402
from vnstock.utils.schema import PredictionContext  # noqa: E402


OUTPUT_PATH = ROOT / "outputs" / "final" / "kronos_full_test_predictions.parquet"
REPORT_DIR = ROOT / "outputs" / "reports" / "kronos_full_test"
PROGRESS_PATH = REPORT_DIR / "progress.json"
LOG_PATH = REPORT_DIR / "kronos_full_test.log"


def main() -> None:
    ensure_dir(REPORT_DIR)
    _seed_everything(42)
    _log("Loading Kronos model...")
    module = importlib.import_module("model")
    tokenizer = module.KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = module.Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = module.KronosPredictor(model, tokenizer, device="cpu", max_context=64)

    feature_panel = load_shared_feature_panel()
    raw_panel = load_table(path_for("interim_root") / "cleaned_panel.parquet")
    feature_panel = feature_panel.loc[feature_panel["split"].astype(str) == "test"].copy()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    raw_panel = raw_panel.sort_values(["symbol", "date"]).copy()
    raw_panel["date"] = pd.to_datetime(raw_panel["date"])

    symbols = sorted(feature_panel["symbol"].astype(str).unique())
    completed = _completed_symbols()
    records = _load_existing_records()
    start = time.time()
    _write_progress(symbols, completed, len(records), start, status="running")

    for symbol_index, symbol in enumerate(symbols, start=1):
        if symbol in completed:
            continue
        symbol_start = time.time()
        symbol_records = _predict_symbol(
            predictor=predictor,
            symbol=symbol,
            feature_panel=feature_panel,
            raw_panel=raw_panel,
        )
        records.extend(symbol_records)
        completed.add(symbol)
        _save_predictions(records)
        elapsed_symbol = time.time() - symbol_start
        elapsed_total = time.time() - start
        _log(
            f"{symbol_index}/{len(symbols)} {symbol}: rows={len(symbol_records)} "
            f"symbol_seconds={elapsed_symbol:.1f} total_rows={len(records)} elapsed_seconds={elapsed_total:.1f}"
        )
        _write_progress(symbols, completed, len(records), start, status="running")

    _write_progress(symbols, completed, len(records), start, status="completed")
    _log(f"Completed full-test Kronos: rows={len(records)} output={OUTPUT_PATH}")


def _predict_symbol(*, predictor: Any, symbol: str, feature_panel: pd.DataFrame, raw_panel: pd.DataFrame) -> list[dict[str, Any]]:
    symbol_features = feature_panel.loc[feature_panel["symbol"].astype(str) == symbol].copy()
    symbol_raw = raw_panel.loc[raw_panel["symbol"].astype(str) == symbol].reset_index(drop=True)
    raw_date_to_index = {date: idx for idx, date in enumerate(symbol_raw["date"])}

    records: list[dict[str, Any]] = []
    for _, row in symbol_features.iterrows():
        raw_idx = raw_date_to_index.get(row["date"])
        if raw_idx is None or raw_idx < 63:
            continue
        if raw_idx + 5 >= len(symbol_raw):
            continue
        context = symbol_raw.iloc[raw_idx - 63 : raw_idx + 1]
        future = symbol_raw.iloc[raw_idx + 1 : raw_idx + 6]
        pred_df = predictor.predict(
            df=context[["open", "high", "low", "close", "volume"]].reset_index(drop=True),
            x_timestamp=context["date"].reset_index(drop=True),
            y_timestamp=future["date"].reset_index(drop=True),
            pred_len=5,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        predicted_close = float(pred_df["close"].iloc[-1])
        current_close = float(row["close"])
        records.append(
            {
                "symbol": symbol,
                "date": row["date"],
                "split": "test",
                "target_ret_5d": float(row["target_ret_5d"]),
                "y_pred": predicted_close / current_close - 1.0,
            }
        )
    return records


def _save_predictions(records: list[dict[str, Any]]) -> None:
    raw = pd.DataFrame(records)
    context = PredictionContext(
        model_family="Kronos zero-shot full-test",
        model_version="NeoQuasar/Kronos-small",
        target_name="target_ret_5d",
        horizon=5,
        run_id="kronos_full_test_cpu",
    )
    output = standardize_prediction_output(raw, context)
    output["coverage_scope"] = "full_test"
    output["direction_score_column"] = "y_pred"
    output["direction_threshold"] = 0.0
    save_table(output, OUTPUT_PATH)


def _load_existing_records() -> list[dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return []
    frame = load_table(OUTPUT_PATH)
    return [
        {
            "symbol": row.symbol,
            "date": row.date,
            "split": row.split,
            "target_ret_5d": float(row.y_true),
            "y_pred": float(row.y_pred),
        }
        for row in frame.itertuples(index=False)
    ]


def _completed_symbols() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    frame = load_table(OUTPUT_PATH)
    return set(frame["symbol"].astype(str).unique())


def _write_progress(symbols: list[str], completed: set[str], rows: int, start: float, *, status: str) -> None:
    elapsed = time.time() - start
    remaining = max(len(symbols) - len(completed), 0)
    rate = len(completed) / elapsed if elapsed > 0 else 0.0
    eta_seconds = remaining / rate if rate > 0 else None
    payload = {
        "status": status,
        "symbols_total": len(symbols),
        "symbols_completed": len(completed),
        "symbols_remaining": remaining,
        "rows": rows,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds,
        "output_path": str(OUTPUT_PATH),
    }
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _log(message: str) -> None:
    ensure_dir(LOG_PATH.parent)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
