from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vnstock.evaluation.metrics import (  # noqa: E402
    balanced_directional_accuracy,
    directional_accuracy,
    f1_down,
    f1_up,
    majority_baseline_accuracy,
)
from vnstock.pipelines.run_investment_backtest import _markdown_table, compute_cross_section_stats  # noqa: E402
from vnstock.utils.io import ensure_dir, load_table, save_table  # noqa: E402


PREDICTIONS = ROOT / "outputs" / "final" / "kronos_full_test_predictions.parquet"
REPORT_DIR = ROOT / "outputs" / "reports" / "kronos_full_test"
PROGRESS = REPORT_DIR / "progress.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Kronos full-test metrics after long zero-shot inference.")
    parser.add_argument("--wait", action="store_true", help="Wait until progress.json reports completed.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wait:
        wait_for_completion(args.poll_seconds)
    metrics = compute_metrics(args.top_k)
    ensure_dir(REPORT_DIR)
    metrics_path = save_table(pd.DataFrame([metrics]), REPORT_DIR / "kronos_full_test_metrics.csv")
    report_path = write_report(metrics, REPORT_DIR / "kronos_full_test_metrics.md", metrics_path=metrics_path)
    print(metrics_path)
    print(report_path)


def wait_for_completion(poll_seconds: int) -> None:
    while True:
        if PROGRESS.exists():
            progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
            if progress.get("status") == "completed":
                return
        time.sleep(max(5, poll_seconds))


def compute_metrics(top_k: int) -> dict[str, object]:
    frame = load_table(PREDICTIONS)
    test = frame.loc[frame["split"].astype(str) == "test"].dropna(subset=["date", "symbol", "y_true", "y_pred"]).copy()
    test["date"] = pd.to_datetime(test["date"])
    cross_section = compute_cross_section_stats(test, score_column="y_pred", top_k=top_k)
    y_true = test["y_true"].to_numpy(dtype=float)
    y_score = test["y_pred"].to_numpy(dtype=float)
    return {
        "model": "Kronos zero-shot full-test",
        "rows": int(len(test)),
        "dates": int(test["date"].nunique()) if not test.empty else 0,
        "symbols": int(test["symbol"].nunique()) if not test.empty else 0,
        "IC": cross_section["IC"],
        "RankIC": cross_section["RankIC"],
        "ICIR": cross_section["ICIR"],
        "RankICIR": cross_section["RankICIR"],
        "Direction_Acc": directional_accuracy(y_true, y_score, 0.0) if len(test) else math.nan,
        "Balanced_Acc": balanced_directional_accuracy(y_true, y_score, 0.0) if len(test) else math.nan,
        "Majority": majority_baseline_accuracy(y_true) if len(test) else math.nan,
        "F1_up": f1_up(y_true, y_score, 0.0) if len(test) else math.nan,
        "F1_down": f1_down(y_true, y_score, 0.0) if len(test) else math.nan,
        "Top5_Return": cross_section["TopK_Return"],
        "Bottom5_Return": cross_section["BottomK_Return"],
        "LongShort5": cross_section["LongShort"],
        "Top5_Direction_Acc": cross_section["TopK_Direction_Acc"],
    }


def write_report(metrics: dict[str, object], output_path: Path, *, metrics_path: Path) -> Path:
    lines = [
        "# Kronos Full-Test Metrics",
        "",
        _markdown_table(pd.DataFrame([metrics])),
        "",
        "## Artifact",
        "",
        f"- Metrics CSV: `{metrics_path.as_posix()}`",
        f"- Predictions: `{PREDICTIONS.as_posix()}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    main()
