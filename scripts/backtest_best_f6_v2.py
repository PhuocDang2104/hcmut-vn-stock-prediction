from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--predictions",
        str(ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"),
        *sys.argv[1:],
    ]
    runpy.run_module("vnstock.pipelines.run_investment_backtest", run_name="__main__")
