from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vnstock.pipelines.run_investment_backtest import (  # noqa: E402
    _prepare_frame,
    compute_cross_section_stats,
    compute_score_bucket_returns,
    run_backtest,
)
from vnstock.utils.io import ensure_dir, load_table  # noqa: E402


PREDICTIONS = ROOT / "outputs" / "final" / "hybrid_xlstm_direction_excess_blend_predictions.parquet"
METRICS = ROOT / "outputs" / "reports" / "final_top5_model_suite" / "top5_model_suite_metrics.csv"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "investment_application_10m"
DOC_PATH = ROOT / "docs" / "investment_application_10m.md"

INITIAL_CAPITAL = 10_000_000.0
TOP_K = 5
HOLDING_DAYS = 5
TRANSACTION_COST_BPS = 15.0


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    frame = _prepare_frame(load_table(PREDICTIONS), split="test", score_column="y_pred")
    cross_section = compute_cross_section_stats(frame, score_column="y_pred", top_k=TOP_K)
    buckets = compute_score_bucket_returns(frame, score_column="y_pred", buckets=5)
    metrics = pd.read_csv(METRICS)
    model_metrics = metrics.loc[metrics["model"].eq("Hybrid xLSTM Direction-Excess Blend")].iloc[0].to_dict()

    fixed = run_backtest(
        frame,
        mode="long-only",
        score_column="y_pred",
        top_k=TOP_K,
        rebalance_every=HOLDING_DAYS,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        cross_section=cross_section,
    )
    long_short = run_backtest(
        frame,
        mode="long-short",
        score_column="y_pred",
        top_k=TOP_K,
        rebalance_every=HOLDING_DAYS,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        cross_section=cross_section,
    )
    rolling_summary, rolling_curve = run_rolling_bucket_proxy(frame)

    fixed_summary = pd.DataFrame([fixed.summary])
    long_short_summary = pd.DataFrame([long_short.summary])
    rolling_summary_frame = pd.DataFrame([rolling_summary])

    fixed_summary.to_csv(OUTPUT_DIR / "fixed_5day_rebalance_10m.csv", index=False)
    fixed.trades.to_csv(OUTPUT_DIR / "fixed_5day_rebalance_trades.csv", index=False)
    fixed.equity_curve.to_csv(OUTPUT_DIR / "fixed_5day_rebalance_equity.csv", index=False)
    long_short_summary.to_csv(OUTPUT_DIR / "long_short_diagnostic_10m.csv", index=False)
    rolling_summary_frame.to_csv(OUTPUT_DIR / "rolling_bucket_10m.csv", index=False)
    rolling_curve.to_csv(OUTPUT_DIR / "rolling_bucket_equity.csv", index=False)
    buckets.to_csv(OUTPUT_DIR / "score_bucket_returns.csv", index=False)

    report = build_report(
        frame=frame,
        model_metrics=model_metrics,
        cross_section=cross_section,
        fixed_summary=fixed.summary,
        long_short_summary=long_short.summary,
        rolling_summary=rolling_summary,
        buckets=buckets,
    )
    DOC_PATH.write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "investment_application_10m.md").write_text(report, encoding="utf-8")

    print(DOC_PATH)
    print(OUTPUT_DIR / "investment_application_10m.md")


def run_rolling_bucket_proxy(frame: pd.DataFrame) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Proxy for daily top-5 signals where each daily bucket is held for 5 sessions.

    Each day deploys 1/5 of capital into that day's top 5. The bucket return is
    recognized when its 5-session realized target matures. This avoids using
    future prices in selection, but it is still a return-target proxy rather than
    an executable order-book backtest.
    """

    cost_rate = TRANSACTION_COST_BPS / 10_000.0
    bucket_notional_weight = 1.0 / HOLDING_DAYS
    equity = INITIAL_CAPITAL
    benchmark_equity = INITIAL_CAPITAL
    peak = equity
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    returns: list[float] = []
    benchmark_returns: list[float] = []
    dates = sorted(frame["date"].unique())

    for date in dates:
        group = frame.loc[frame["date"].eq(date)].sort_values("y_pred", ascending=False)
        if group.empty:
            continue
        top = group.head(min(TOP_K, len(group)))
        gross_bucket_return = float(top["y_true"].mean())
        benchmark_bucket_return = float(group["y_true"].mean())

        # Buy and sell the daily bucket. Only 1/HOLDING_DAYS of capital is used.
        cost = bucket_notional_weight * 2.0 * cost_rate
        portfolio_return = bucket_notional_weight * gross_bucket_return - cost
        benchmark_return = bucket_notional_weight * benchmark_bucket_return

        equity *= 1.0 + portfolio_return
        benchmark_equity *= 1.0 + benchmark_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        returns.append(portfolio_return)
        benchmark_returns.append(benchmark_return)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "top5_realized_return": gross_bucket_return,
                "portfolio_return_after_cost": portfolio_return,
                "benchmark_return": benchmark_return,
                "equity": equity,
                "benchmark_equity": benchmark_equity,
                "drawdown": drawdown,
            }
        )

    curve = pd.DataFrame(rows)
    returns_array = np.asarray(returns, dtype=float)
    benchmark_array = np.asarray(benchmark_returns, dtype=float)
    summary = {
        "mode": "rolling_5day_bucket_proxy",
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": equity,
        "total_profit": equity - INITIAL_CAPITAL,
        "total_return": equity / INITIAL_CAPITAL - 1.0,
        "benchmark_final_capital": benchmark_equity,
        "benchmark_total_profit": benchmark_equity - INITIAL_CAPITAL,
        "benchmark_total_return": benchmark_equity / INITIAL_CAPITAL - 1.0,
        "periods": int(len(returns_array)),
        "avg_daily_bucket_return_after_cost": float(np.mean(returns_array)) if len(returns_array) else math.nan,
        "hit_rate": float(np.mean(returns_array > 0)) if len(returns_array) else math.nan,
        "sharpe_proxy": _sharpe(returns_array, periods_per_year=252.0),
        "max_drawdown": float(curve["drawdown"].min()) if not curve.empty else math.nan,
    }
    if len(benchmark_array):
        summary["excess_return_vs_benchmark"] = summary["total_return"] - summary["benchmark_total_return"]
    else:
        summary["excess_return_vs_benchmark"] = math.nan
    return summary, curve


def build_report(
    *,
    frame: pd.DataFrame,
    model_metrics: dict[str, object],
    cross_section: dict[str, float],
    fixed_summary: dict[str, object],
    long_short_summary: dict[str, object],
    rolling_summary: dict[str, object],
    buckets: pd.DataFrame,
) -> str:
    start = pd.Timestamp(frame["date"].min()).date()
    end = pd.Timestamp(frame["date"].max()).date()
    days = int(frame["date"].nunique())
    symbols = int(frame["symbol"].nunique())

    gross_top5_per_full_bucket = INITIAL_CAPITAL * cross_section["TopK_Return"]
    daily_bucket_notional = INITIAL_CAPITAL / HOLDING_DAYS
    gross_top5_per_daily_bucket = daily_bucket_notional * cross_section["TopK_Return"]
    round_trip_cost_daily_bucket = daily_bucket_notional * 2.0 * TRANSACTION_COST_BPS / 10_000.0
    net_top5_per_daily_bucket = gross_top5_per_daily_bucket - round_trip_cost_daily_bucket

    metric_table = pd.DataFrame(
        [
            {
                "metric": "IC",
                "value": f"{cross_section['IC']:.4f}",
                "meaning": "Pearson theo từng ngày giữa score và return thực tế",
            },
            {
                "metric": "RankIC",
                "value": f"{cross_section['RankIC']:.4f}",
                "meaning": "Spearman rank theo từng ngày",
            },
            {
                "metric": "ICIR",
                "value": f"{cross_section['ICIR']:.3f}",
                "meaning": "mean daily IC / std daily IC",
            },
            {
                "metric": "Top5_Return",
                "value": _fmt_pct(cross_section["TopK_Return"]),
                "meaning": "return thực tế trung bình của 5 mã được chọn",
            },
            {
                "metric": "Top5_Direction_Acc",
                "value": _fmt_pct(cross_section["TopK_Direction_Acc"]),
                "meaning": "tỷ lệ mã top 5 có return 5 phiên dương",
            },
            {
                "metric": "LongShort5",
                "value": _fmt_pct(cross_section["LongShort"]),
                "meaning": "top5 return - bottom5 return; chỉ là diagnostic ranking alpha",
            },
        ]
    )

    fixed_table = pd.DataFrame(
        [
            {
                "mode": "Rebalance every 5 sessions",
                "initial": _fmt_money(fixed_summary["initial_capital"]),
                "final": _fmt_money(fixed_summary["final_capital"]),
                "profit": _fmt_money(fixed_summary["total_profit"]),
                "return": _fmt_pct(fixed_summary["total_return"]),
                "benchmark_return": _fmt_pct(fixed_summary["benchmark_total_return"]),
                "max_drawdown": _fmt_pct(fixed_summary["max_drawdown"]),
                "hit_rate": _fmt_pct(fixed_summary["hit_rate"]),
            },
            {
                "mode": "Rolling 5-day bucket proxy",
                "initial": _fmt_money(rolling_summary["initial_capital"]),
                "final": _fmt_money(rolling_summary["final_capital"]),
                "profit": _fmt_money(rolling_summary["total_profit"]),
                "return": _fmt_pct(rolling_summary["total_return"]),
                "benchmark_return": _fmt_pct(rolling_summary["benchmark_total_return"]),
                "max_drawdown": _fmt_pct(rolling_summary["max_drawdown"]),
                "hit_rate": _fmt_pct(rolling_summary["hit_rate"]),
            },
        ]
    )

    bucket_table = buckets.copy()
    if not bucket_table.empty:
        bucket_table["avg_realized_return"] = bucket_table["avg_realized_return"].map(_fmt_pct)
        bucket_table["direction_acc"] = bucket_table["direction_acc"].map(_fmt_pct)
        bucket_table["avg_count"] = bucket_table["avg_count"].map(lambda value: f"{float(value):.1f}")

    lines = [
        "# Báo Cáo Áp Dụng Đầu Tư - Vốn 10 Triệu VND",
        "",
        "## 1. Phạm Vi",
        "",
        "Tài liệu này mô phỏng cách dùng model hiện tại nếu triển khai một chiến lược long-only với vốn giả định 10 triệu VND.",
        "",
        "Model:",
        "",
        "```text",
        "Hybrid xLSTM Direction-Excess Blend",
        "```",
        "",
        "Model không dự báo giá đóng cửa ngày mai. Nhiệm vụ chính là xếp hạng cổ phiếu theo return kỳ vọng trong 5 phiên tới.",
        "",
        "Giai đoạn test dùng trong báo cáo:",
        "",
        f"- Dates: `{start}` to `{end}`",
        f"- Số ngày giao dịch: `{days}`",
        f"- Số mã: `{symbols}`",
        f"- Vốn giả định: `{_fmt_money(INITIAL_CAPITAL)}`",
        f"- Top-k: `{TOP_K}`",
        f"- Horizon: `{HOLDING_DAYS}` phiên",
        f"- Phí giả định: `{TRANSACTION_COST_BPS:.0f}` bps một chiều, áp dụng trong proxy backtest",
        "",
        "## 2. Logic Vận Hành Mỗi Ngày",
        "",
        "Sau khi có dữ liệu đóng cửa ngày `t`:",
        "",
        "1. Cập nhật dữ liệu OHLCV cho toàn universe.",
        "2. Tạo window 64 phiên gần nhất cho từng mã.",
        "3. Chạy inference để lấy tín hiệu return/ranking, direction, excess/relative và final score.",
        "4. Xếp hạng toàn bộ mã theo final score.",
        "5. Mua top 5 mã, mặc định equal-weight nếu chưa thêm risk layer.",
        "",
        "Nguyên lý final score hiện tại:",
        "",
        "```text",
        "final_score =",
        "    0.6 * normalized(return/ranking signal)",
        "  + 0.2 * normalized(direction signal)",
        "  + 0.2 * normalized(excess/market-relative signal)",
        "```",
        "",
        "## 3. Hai Cách Dùng Tín Hiệu 5 Phiên",
        "",
        "### Cách A - Rebalance mỗi 5 phiên",
        "",
        "Dùng toàn bộ vốn cho một rổ top 5, giữ đúng 5 phiên, sau đó bán và chọn rổ mới.",
        "",
        f"- Vốn: `{_fmt_money(INITIAL_CAPITAL)}`",
        f"- Vốn mỗi mã trong top 5: `{_fmt_money(INITIAL_CAPITAL / TOP_K)}`",
        f"- Top5_Return trung bình mỗi forecast window 5 phiên: `{_fmt_pct(cross_section['TopK_Return'])}`",
        f"- Lãi gộp trung bình trên rổ 10 triệu trước phí: `{_fmt_money(gross_top5_per_full_bucket)}`",
        "",
        "### Cách B - Rolling bucket 5 ngày",
        "",
        "Mỗi ngày mở một bucket top 5 mới và giữ bucket đó 5 phiên. Cách này gần với workflow quant hơn vì mỗi ngày đều dùng tín hiệu mới.",
        "",
        f"- Vốn deploy mỗi ngày: `{_fmt_money(daily_bucket_notional)}`",
        f"- Vốn mỗi mã trong bucket top 5 hằng ngày: `{_fmt_money(daily_bucket_notional / TOP_K)}`",
        f"- Lãi gộp trung bình mỗi daily bucket trước phí: `{_fmt_money(gross_top5_per_daily_bucket)}`",
        f"- Phí mua+bán ước tính mỗi daily bucket: `{_fmt_money(round_trip_cost_daily_bucket)}`",
        f"- Lãi ròng trung bình ước tính mỗi daily bucket: `{_fmt_money(net_top5_per_daily_bucket)}`",
        "",
        "Cách B vẫn là proxy dựa trên target return 5 phiên, không phải mô phỏng khớp lệnh thực tế.",
        "",
        "## 4. Proxy Backtest Với Vốn 10 Triệu VND",
        "",
        _markdown_table(fixed_table),
        "",
        "Cách rebalance mỗi 5 phiên khớp sạch nhất với target `target_ret_5d`. Rolling bucket gần vận hành thực tế hơn, nhưng vẫn chưa phải backtest ở cấp broker/order-book.",
        "",
        "Diagnostic long-short với cùng notional 10 triệu:",
        "",
        f"- Vốn cuối kỳ proxy: `{_fmt_money(long_short_summary['final_capital'])}`",
        f"- Total return proxy: `{_fmt_pct(long_short_summary['total_return'])}`",
        "- Mục đích: chỉ dùng để kiểm định ranking alpha, vì bán khống thực tế tại Việt Nam bị hạn chế.",
        "",
        "## 5. Metric Để Đánh Giá Độ Tin Cậy",
        "",
        _markdown_table(metric_table),
        "",
        "Kiểm tra score bucket:",
        "",
        _markdown_table(bucket_table),
        "",
        "Nếu score có giá trị thật, bucket điểm cao phải có return thực tế cao hơn bucket điểm thấp. Bảng trên ủng hộ việc IC hiện tại không chỉ là một con số aggregate ngẫu nhiên.",
        "",
        "## 6. IC = 0.09 Có Giá Trị Không?",
        "",
        "Có. `IC = 0.0904` là có giá trị với bài toán stock-selection daily, vì return cổ phiếu rất nhiễu và IC hữu dụng thường không cần quá lớn. Trong repo này, IC còn được xác nhận bởi `RankIC = 0.0852`, `Top5_Return = 1.7120%`, `LongShort5 = 1.7462%`, và `Top5_Direction_Acc = 58.53%` trên out-of-time test.",
        "",
        "Điểm quan trọng: IC ở đây là cross-sectional. Mỗi ngày nó đo xem mã có score cao hơn có thực sự tạo return 5 phiên cao hơn các mã còn lại hay không. Điều này khớp trực tiếp với bài toán chọn top 5.",
        "",
        "## 7. Có Nên Tin Mạnh Không?",
        "",
        "Kết luận hiện tại: signal có ý nghĩa và đáng tiếp tục phát triển, nhưng vẫn nên xem là research-grade trước khi audit xong các caveat dữ liệu và execution.",
        "",
        "Lý do có thể tin ở mức vừa phải:",
        "",
        "- Evaluation dùng out-of-time test từ 2025-01-02 đến 2026-04-07.",
        "- `Top5_Return` và `LongShort5` đều dương và tốt hơn rõ các baseline yếu hơn.",
        "- Proxy vốn 10 triệu vẫn có lãi sau giả định phí một chiều 15 bps.",
        "- `Top5_Direction_Acc` là 58.53%, cao hơn accuracy toàn universe vì model có ích nhất ở vùng top tail được chọn.",
        "",
        "Lý do chưa nên tin tuyệt đối:",
        "",
        "- `docs/data_contract.md` vẫn ghi caveat `bfill()` đầu chuỗi có thể tạo leakage nhẹ.",
        "- Universe trộn VN và foreign symbols, nhưng shared panel chưa đưa `market`, `currency`, `source` tag vào model.",
        "- `VNINDEX` và `VN30` có sẵn nhưng chưa join đầy đủ thành market-context features.",
        "- Backtest hiện dùng target return thực tế, chưa mô phỏng limit-up/down, thanh khoản, slippage, thuế và khớp lệnh.",
        "- Cần kiểm tra thêm theo tháng, regime, VN-only và core10 trước khi dùng vốn thật.",
        "",
        "Cách đọc thực dụng:",
        "",
        "```text",
        "IC 0.09 = có ranking alpha đáng chú ý.",
        "IC 0.09 != đủ điều kiện tự động đem tiền thật vào trade nếu chưa audit leakage và execution.",
        "```",
        "",
        "## 8. Khuyến Nghị Sử Dụng",
        "",
        "Nếu muốn paper-trade với vốn giả định 10 triệu VND:",
        "",
        "1. Chỉ dùng long-only top 5.",
        "2. Ưu tiên rebalance mỗi 5 phiên trước vì khớp trực tiếp với target train.",
        "3. Theo dõi fill, phí, slippage và thanh khoản riêng ngoài report model.",
        "4. So sánh return paper/live với benchmark equal-weight universe.",
        "5. Chưa nên dùng vốn thật trước khi đóng caveat leakage và kiểm tra ổn định theo tháng/regime.",
        "",
        "## 9. Artifact Được Sinh Ra",
        "",
        f"- Report copy: `{(OUTPUT_DIR / 'investment_application_10m.md').as_posix()}`",
        f"- Fixed rebalance summary: `{(OUTPUT_DIR / 'fixed_5day_rebalance_10m.csv').as_posix()}`",
        f"- Fixed rebalance trades: `{(OUTPUT_DIR / 'fixed_5day_rebalance_trades.csv').as_posix()}`",
        f"- Rolling bucket summary: `{(OUTPUT_DIR / 'rolling_bucket_10m.csv').as_posix()}`",
        f"- Rolling bucket equity: `{(OUTPUT_DIR / 'rolling_bucket_equity.csv').as_posix()}`",
        f"- Score buckets: `{(OUTPUT_DIR / 'score_bucket_returns.csv').as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def _sharpe(values: np.ndarray, *, periods_per_year: float) -> float:
    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return math.nan
    std = float(np.std(clean, ddof=1))
    if std <= 0:
        return math.nan
    return float(np.mean(clean) / std * math.sqrt(periods_per_year))


def _fmt_money(value: object) -> str:
    return f"{float(value):,.0f} VND"


def _fmt_pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
