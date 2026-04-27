# Báo Cáo Áp Dụng Đầu Tư - Vốn 10 Triệu VND

## 1. Phạm Vi

Tài liệu này mô phỏng cách dùng model hiện tại nếu triển khai một chiến lược long-only với vốn giả định 10 triệu VND.

Model:

```text
Hybrid xLSTM Direction-Excess Blend
```

Model không dự báo giá đóng cửa ngày mai. Nhiệm vụ chính là xếp hạng cổ phiếu theo return kỳ vọng trong 5 phiên tới.

Giai đoạn test dùng trong báo cáo:

- Dates: `2025-01-02` to `2026-04-07`
- Số ngày giao dịch: `327`
- Số mã: `95`
- Vốn giả định: `10,000,000 VND`
- Top-k: `5`
- Horizon: `5` phiên
- Phí giả định: `15` bps một chiều, áp dụng trong proxy backtest

## 2. Logic Vận Hành Mỗi Ngày

Sau khi có dữ liệu đóng cửa ngày `t`:

1. Cập nhật dữ liệu OHLCV cho toàn universe.
2. Tạo window 64 phiên gần nhất cho từng mã.
3. Chạy inference để lấy tín hiệu return/ranking, direction, excess/relative và final score.
4. Xếp hạng toàn bộ mã theo final score.
5. Mua top 5 mã, mặc định equal-weight nếu chưa thêm risk layer.

Nguyên lý final score hiện tại:

```text
final_score =
    0.6 * normalized(return/ranking signal)
  + 0.2 * normalized(direction signal)
  + 0.2 * normalized(excess/market-relative signal)
```

## 3. Hai Cách Dùng Tín Hiệu 5 Phiên

### Cách A - Rebalance mỗi 5 phiên

Dùng toàn bộ vốn cho một rổ top 5, giữ đúng 5 phiên, sau đó bán và chọn rổ mới.

- Vốn: `10,000,000 VND`
- Vốn mỗi mã trong top 5: `2,000,000 VND`
- Top5_Return trung bình mỗi forecast window 5 phiên: `1.71%`
- Lãi gộp trung bình trên rổ 10 triệu trước phí: `171,201 VND`

### Cách B - Rolling bucket 5 ngày

Mỗi ngày mở một bucket top 5 mới và giữ bucket đó 5 phiên. Cách này gần với workflow quant hơn vì mỗi ngày đều dùng tín hiệu mới.

- Vốn deploy mỗi ngày: `2,000,000 VND`
- Vốn mỗi mã trong bucket top 5 hằng ngày: `400,000 VND`
- Lãi gộp trung bình mỗi daily bucket trước phí: `34,240 VND`
- Phí mua+bán ước tính mỗi daily bucket: `6,000 VND`
- Lãi ròng trung bình ước tính mỗi daily bucket: `28,240 VND`

Cách B vẫn là proxy dựa trên target return 5 phiên, không phải mô phỏng khớp lệnh thực tế.

## 4. Proxy Backtest Với Vốn 10 Triệu VND

| mode | initial | final | profit | return | benchmark_return | max_drawdown | hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rebalance every 5 sessions | 10,000,000 VND | 23,761,800 VND | 13,761,800 VND | 137.62% | 25.53% | -22.25% | 62.12% |
| Rolling 5-day bucket proxy | 10,000,000 VND | 24,836,924 VND | 14,836,924 VND | 148.37% | 41.43% | -16.09% | 63.61% |

Cách rebalance mỗi 5 phiên khớp sạch nhất với target `target_ret_5d`. Rolling bucket gần vận hành thực tế hơn, nhưng vẫn chưa phải backtest ở cấp broker/order-book.

Diagnostic long-short với cùng notional 10 triệu:

- Vốn cuối kỳ proxy: `27,993,789 VND`
- Total return proxy: `179.94%`
- Mục đích: chỉ dùng để kiểm định ranking alpha, vì bán khống thực tế tại Việt Nam bị hạn chế.

## 5. Metric Để Đánh Giá Độ Tin Cậy

| metric | value | meaning |
| --- | --- | --- |
| IC | 0.0904 | Pearson theo từng ngày giữa score và return thực tế |
| RankIC | 0.0852 | Spearman rank theo từng ngày |
| ICIR | 0.465 | mean daily IC / std daily IC |
| Top5_Return | 1.71% | return thực tế trung bình của 5 mã được chọn |
| Top5_Direction_Acc | 58.53% | tỷ lệ mã top 5 có return 5 phiên dương |
| LongShort5 | 1.75% | top5 return - bottom5 return; chỉ là diagnostic ranking alpha |

Kiểm tra score bucket:

| score_bucket | avg_realized_return | direction_acc | avg_count |
| --- | --- | --- | --- |
| 1 | -0.01% | 47.05% | 18.2 |
| 2 | 0.27% | 50.13% | 18.1 |
| 3 | 0.44% | 52.03% | 18.0 |
| 4 | 0.68% | 54.50% | 18.1 |
| 5 | 1.27% | 57.06% | 18.2 |

Nếu score có giá trị thật, bucket điểm cao phải có return thực tế cao hơn bucket điểm thấp. Bảng trên ủng hộ việc IC hiện tại không chỉ là một con số aggregate ngẫu nhiên.

## 6. IC = 0.09 Có Giá Trị Không?

Có. `IC = 0.0904` là có giá trị với bài toán stock-selection daily, vì return cổ phiếu rất nhiễu và IC hữu dụng thường không cần quá lớn. Trong repo này, IC còn được xác nhận bởi `RankIC = 0.0852`, `Top5_Return = 1.7120%`, `LongShort5 = 1.7462%`, và `Top5_Direction_Acc = 58.53%` trên out-of-time test.

Điểm quan trọng: IC ở đây là cross-sectional. Mỗi ngày nó đo xem mã có score cao hơn có thực sự tạo return 5 phiên cao hơn các mã còn lại hay không. Điều này khớp trực tiếp với bài toán chọn top 5.

## 7. Có Nên Tin Mạnh Không?

Kết luận hiện tại: signal có ý nghĩa và đáng tiếp tục phát triển, nhưng vẫn nên xem là research-grade trước khi audit xong các caveat dữ liệu và execution.

Lý do có thể tin ở mức vừa phải:

- Evaluation dùng out-of-time test từ 2025-01-02 đến 2026-04-07.
- `Top5_Return` và `LongShort5` đều dương và tốt hơn rõ các baseline yếu hơn.
- Proxy vốn 10 triệu vẫn có lãi sau giả định phí một chiều 15 bps.
- `Top5_Direction_Acc` là 58.53%, cao hơn accuracy toàn universe vì model có ích nhất ở vùng top tail được chọn.

Lý do chưa nên tin tuyệt đối:

- `docs/data_contract.md` vẫn ghi caveat `bfill()` đầu chuỗi có thể tạo leakage nhẹ.
- Universe trộn VN và foreign symbols, nhưng shared panel chưa đưa `market`, `currency`, `source` tag vào model.
- `VNINDEX` và `VN30` có sẵn nhưng chưa join đầy đủ thành market-context features.
- Backtest hiện dùng target return thực tế, chưa mô phỏng limit-up/down, thanh khoản, slippage, thuế và khớp lệnh.
- Cần kiểm tra thêm theo tháng, regime, VN-only và core10 trước khi dùng vốn thật.

Cách đọc thực dụng:

```text
IC 0.09 = có ranking alpha đáng chú ý.
IC 0.09 != đủ điều kiện tự động đem tiền thật vào trade nếu chưa audit leakage và execution.
```

## 8. Khuyến Nghị Sử Dụng

Nếu muốn paper-trade với vốn giả định 10 triệu VND:

1. Chỉ dùng long-only top 5.
2. Ưu tiên rebalance mỗi 5 phiên trước vì khớp trực tiếp với target train.
3. Theo dõi fill, phí, slippage và thanh khoản riêng ngoài report model.
4. So sánh return paper/live với benchmark equal-weight universe.
5. Chưa nên dùng vốn thật trước khi đóng caveat leakage và kiểm tra ổn định theo tháng/regime.

## 9. Artifact Được Sinh Ra

- Report copy: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/investment_application_10m/investment_application_10m.md`
- Fixed rebalance summary: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/investment_application_10m/fixed_5day_rebalance_10m.csv`
- Fixed rebalance trades: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/investment_application_10m/fixed_5day_rebalance_trades.csv`
- Rolling bucket summary: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/investment_application_10m/rolling_bucket_10m.csv`
- Rolling bucket equity: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/investment_application_10m/rolling_bucket_equity.csv`
- Score buckets: `C:/Users/ADMIN/Desktop/vn-stock-prediction/outputs/reports/investment_application_10m/score_bucket_returns.csv`
