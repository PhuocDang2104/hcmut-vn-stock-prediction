# Final Model Architecture And Principles

Tài liệu này mô tả bộ model cuối đang dùng để so sánh trên bài toán stock-selection top 5, horizon `t+5`, universe `95` mã. Mục tiêu chính không phải dự báo chính xác từng giá đóng cửa, mà là xếp hạng mã cổ phiếu tại ngày `t` để chọn danh mục top 5 có realized return tốt trong `5` phiên tiếp theo.

## 1. Bài Toán Và Contract Chung

Target chính:

```text
target_ret_5d = close[t+5] / close[t] - 1
```

Input chuẩn cho neural models:

```text
X shape = [batch, seq_len, num_features]
seq_len = 64
num_features = 33
```

Input của tabular model:

```text
X shape = [n_rows, num_features]
num_features = 33
```

Các feature đang dùng trong final suite:

```text
ret_1d, ret_3d, ret_5d, ret_10d, ret_20d,
log_volume, volume_zscore_20,
hl_spread, oc_change, gap_rel,
body_ratio, upper_shadow_ratio, lower_shadow_ratio,
rolling_vol_5, rolling_vol_20, vol_ratio_5_20,
ma_ratio_5_20,
open_rel, high_rel, low_rel, ma5_rel, ma20_rel,
rsi_14, macd_hist,
market_ret_1d, market_ret_5d, market_ret_20d, market_vol_20,
excess_ret_1d, excess_ret_5d, excess_ret_20d,
relative_strength_5d, relative_strength_20d
```

Các metric chính:

| Metric | Ý nghĩa |
| --- | --- |
| `IC` | Pearson correlation theo từng ngày giữa score và realized return, rồi lấy trung bình |
| `RankIC` | Spearman/rank correlation theo từng ngày giữa score rank và realized return rank |
| `ICIR` | mean daily IC / std daily IC |
| `RankICIR` | mean daily RankIC / std daily RankIC |
| `Direction_Acc` | Accuracy dấu tăng/giảm trên toàn bộ sample |
| `Balanced_Acc` | Accuracy cân bằng giữa class up/down |
| `Top5_Return` | Realized return trung bình của 5 mã score cao nhất mỗi ngày |
| `Bottom5_Return` | Realized return trung bình của 5 mã score thấp nhất mỗi ngày |
| `LongShort5` | `Top5_Return - Bottom5_Return` |
| `Top5_Direction_Acc` | Tỷ lệ mã top 5 có realized return dương |

Trong bài toán hiện tại, `IC`, `RankIC`, `Top5_Return` và `LongShort5` quan trọng hơn raw `Direction_Acc`, vì mục tiêu là chọn danh mục tốt hơn phần còn lại, không phải đoán đúng dấu của mọi mã.

## 2. Leakage Controls

Final suite giữ các nguyên tắc sau:

- Split theo thời gian: train tới `2023-12-31`, valid trong `2024`, test từ `2025-01-02` tới `2026-04-07`.
- Scaler fit trên train split, sau đó áp dụng cho valid/test.
- Sequence được build theo từng symbol, không trộn tương lai của mã này vào mã khác.
- Target `target_ret_5d` chỉ dùng `close[t+5]` làm nhãn, không dùng trong input window.
- Feature rolling/momentum là past-only theo từng symbol.
- Kronos chỉ dùng zero-shot inference, không fine-tune trên test.

## 3. Hybrid xLSTM Direction-Excess Blend

Tên kỹ thuật cuối:

```text
Hybrid xLSTM Direction-Excess Blend
```

Tên này thay cho tên thử nghiệm cũ `BestF6-v2`.

### 3.1. Nguyên lý

Model này không chỉ dự báo một return scalar. Nó blend ba nhóm tín hiệu:

```text
final_score =
    0.6 * normalized(return/ranking signal)
  + 0.2 * normalized(direction signal)
  + 0.2 * normalized(excess/market-relative signal)
```

Ý nghĩa:

- `return/ranking signal`: dự đoán return 5 phiên tới, giúp tạo IC và RankIC.
- `direction signal`: xác suất hoặc logit tăng/giảm, giúp ổn định dấu và top-5 direction.
- `excess/market-relative signal`: tín hiệu vượt trội so với thị trường/universe, giúp stock selection và long-short spread.

Đây là model phù hợp nhất với bài toán chọn cổ phiếu vì score cuối được thiết kế để rank cross-section, không chỉ minimize regression loss.

### 3.2. Backbone xLSTM trong repo

Implementation base: `src/vnstock/models/xlstm_ts/model.py`.

Input:

```text
[batch, 64, num_features]
```

Các layer chính:

| Block | Layer | Vai trò |
| --- | --- | --- |
| Input projection | `Linear(num_features, hidden_dim)` | Chiếu feature đầu vào sang latent dimension |
| Residual block | `LayerNorm -> LSTM -> Dropout -> residual` | Học quan hệ temporal trong 64 phiên |
| Feed-forward block | `LayerNorm -> Linear(2x) -> GELU -> Dropout -> Linear -> residual` | Tăng nonlinear capacity sau recurrent block |
| Output norm | `LayerNorm(hidden_dim)` | Ổn định representation cuối |
| Optional pooling | `last` hoặc `gated_concat` | Dùng hidden cuối hoặc concat hidden cuối với gated temporal pooling |
| Return head | `Linear -> GELU -> Dropout -> Linear(1)` | Dự báo return/ranking signal |
| Direction head | `Linear -> GELU -> Dropout -> Linear(1)` | Dự báo direction logit |

Gated pooling nếu bật:

```text
weights_t = softmax(Linear(LayerNorm(h_t)))
gated_pool = sum_t weights_t * h_t
pooled = LayerNorm(concat(last_state, gated_pool))
```

Điểm mạnh của gated pooling là model không bị phụ thuộc hoàn toàn vào ngày cuối trong window; nó học phiên nào trong 64 phiên là quan trọng hơn.

### 3.3. Vì sao blend này tốt nhất

Kết quả test full 95 mã:

| Model | IC | RankIC | Direction Acc | Balanced Acc | Top5 Return | LongShort5 | Top5 Direction Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid xLSTM Direction-Excess Blend | 0.0904 | 0.0852 | 52.65% | 52.41% | 1.7120% | 1.7462% | 58.53% |
| LightGBM-style HGBR | 0.0545 | 0.0379 | 50.53% | 49.89% | 1.2315% | 1.3634% | 55.41% |
| CNN-LSTM | 0.0522 | 0.0433 | 51.80% | 50.32% | 1.2969% | 1.2686% | 56.70% |
| TCN | 0.0212 | 0.0216 | 48.91% | 48.89% | 0.7223% | 0.4266% | 54.68% |
| PatchTST | 0.0185 | 0.0150 | 48.58% | 49.06% | 0.6983% | 0.2568% | 52.78% |
| Kronos zero-shot | 0.0069 | 0.0189 | 50.95% | 51.08% | 0.3581% | -0.1264% | 52.42% |

Hybrid xLSTM thắng vì:

- IC cao nhất: `0.0904`, nghĩa là score có quan hệ tuyến tính tốt nhất với realized return theo từng ngày.
- RankIC cao nhất: `0.0852`, nghĩa là thứ hạng score khớp tốt nhất với thứ hạng return thực tế.
- Top5_Return cao nhất: top 5 mã model chọn có realized return trung bình `1.7120%` mỗi forecast window.
- LongShort5 cao nhất: `1.7462%`, chứng minh model không chỉ chọn được nhóm tốt mà còn đẩy nhóm xấu xuống đáy ranking.
- Top5_Direction_Acc cao nhất: `58.53%`, tốt hơn accuracy toàn universe vì model tập trung đúng hơn ở phần danh mục thật sự được chọn.
- Blend có market-relative/excess signal, trong khi các baseline chủ yếu dự báo return trực tiếp.

Raw `Direction_Acc` chỉ khoảng `52.65%`, nhưng đây không phải điểm yếu lớn trong stock-selection. Full-universe direction accuracy tính cả các sample gần 0% return, vốn rất nhiễu. Quan trọng hơn là model rank đúng vùng cực trị, và điều đó thể hiện ở `IC`, `RankIC`, `Top5_Return`, `LongShort5`.

## 4. CNN-LSTM

Implementation: `scripts/run_final_top5_model_suite.py`.

### 4.1. Nguyên lý

CNN-LSTM kết hợp:

- CNN 1D để bắt pattern local trong chuỗi K-line, ví dụ cụm nến, momentum ngắn hạn, biến động volume.
- LSTM để học dependency theo thời gian sau khi local pattern đã được encode.

### 4.2. Kiến trúc

Input:

```text
[batch, 64, 33]
```

Layer stack:

| Thứ tự | Layer | Parameter |
| ---: | --- | --- |
| 1 | `Conv1d` | `in_channels=33`, `out_channels=64`, `kernel_size=3`, `padding=1` |
| 2 | `GELU` | activation |
| 3 | `Dropout` | `p=0.1` |
| 4 | `Conv1d` | `in_channels=64`, `out_channels=96`, `kernel_size=3`, `padding=1` |
| 5 | `GELU` | activation |
| 6 | `LSTM` | `input_size=96`, `hidden_size=96`, `batch_first=True` |
| 7 | `LayerNorm` | hidden size `96` trên last timestep |
| 8 | Return head | `Linear(96,96) -> GELU -> Dropout(0.1) -> Linear(96,1)` |
| 9 | Direction head | `Linear(96,96) -> GELU -> Dropout(0.1) -> Linear(96,1)` |

Training config:

```text
epochs = 4
batch_size = 512
eval_batch_size = 1024
learning_rate = 5e-4
weight_decay = 1e-4
huber_delta = 0.05
direction_loss_weight = 0.1
patience = 2
device = CPU
```

### 4.3. Nhận xét

CNN-LSTM là neural baseline tốt thứ hai theo `Top5_Direction_Acc` và khá cạnh tranh về `Top5_Return`. Tuy nhiên nó thiếu explicit excess-return blend, nên `IC`, `RankIC` và `LongShort5` thấp hơn Hybrid xLSTM.

## 5. TCN

Implementation: `scripts/run_final_top5_model_suite.py`.

### 5.1. Nguyên lý

TCN dùng convolution theo thời gian với dilation để mở rộng receptive field mà không cần recurrent state. Với dilation `1, 2, 4, 8`, model có thể nhìn pattern ngắn và trung hạn trong window 64 phiên.

### 5.2. Kiến trúc

Input:

```text
[batch, 64, 33]
```

Layer stack:

| Thứ tự | Layer | Parameter |
| ---: | --- | --- |
| 1 | Projection | `Conv1d(33, 96, kernel_size=1)` |
| 2 | TCN block 1 | channels `96`, dilation `1`, kernel `3`, dropout `0.1` |
| 3 | TCN block 2 | channels `96`, dilation `2`, kernel `3`, dropout `0.1` |
| 4 | TCN block 3 | channels `96`, dilation `4`, kernel `3`, dropout `0.1` |
| 5 | TCN block 4 | channels `96`, dilation `8`, kernel `3`, dropout `0.1` |
| 6 | Pooling | last timestep |
| 7 | LayerNorm | hidden size `96` |
| 8 | Return head | `Linear(96,96) -> GELU -> Dropout(0.1) -> Linear(96,1)` |
| 9 | Direction head | `Linear(96,96) -> GELU -> Dropout(0.1) -> Linear(96,1)` |

Mỗi TCN block:

```text
Conv1d -> GELU -> Dropout -> Conv1d -> GELU -> Dropout -> crop -> residual add -> BatchNorm1d
```

Training config giống CNN-LSTM.

### 5.3. Nhận xét

TCN có inductive bias tốt cho chuỗi, nhưng trong run hiện tại metric thấp hơn nhiều. Nguyên nhân thực dụng có thể là training ngắn, target tài chính nhiễu, và TCN direct-return head chưa có direction/excess blend mạnh như Hybrid xLSTM.

## 6. PatchTST

Implementation: `scripts/run_final_top5_model_suite.py`.

### 6.1. Nguyên lý

PatchTST chia chuỗi thời gian thành các patch giống cách Vision Transformer chia ảnh thành patch. Thay vì attention trên từng ngày, model attention trên token đại diện cho một đoạn thời gian.

### 6.2. Kiến trúc

Input:

```text
[batch, 64, 33]
```

Patch config:

```text
patch_len = 16
stride = 8
patch_count = (64 - 16) / 8 + 1 = 7
patch_dim = 16 * 33 = 528
```

Layer stack:

| Thứ tự | Layer | Parameter |
| ---: | --- | --- |
| 1 | Patch extraction | `unfold(time, patch_len=16, stride=8)` |
| 2 | Patch projection | `Linear(528, 96)` |
| 3 | Positional embedding | learnable `[1, 7, 96]` |
| 4 | Transformer encoder | `2` layers |
| 5 | Multi-head attention | `d_model=96`, `n_heads=4` |
| 6 | FFN | `dim_feedforward=384`, activation `GELU` |
| 7 | Pooling | mean over patch tokens |
| 8 | LayerNorm | hidden size `96` |
| 9 | Return head | `Linear(96,96) -> GELU -> Dropout(0.1) -> Linear(96,1)` |
| 10 | Direction head | `Linear(96,96) -> GELU -> Dropout(0.1) -> Linear(96,1)` |

Training config giống CNN-LSTM.

### 6.3. Nhận xét

PatchTST phù hợp khi chuỗi có pattern dài và dữ liệu đủ lớn. Trong run hiện tại, nó yếu hơn vì patch token hóa có thể làm mất tín hiệu ngắn hạn tinh vi của return 5 phiên, và direct-return training chưa đủ align với top-5 ranking objective.

## 7. LightGBM-Style HGBR

Implementation: `scripts/run_final_top5_model_suite.py`.

### 7.1. Nguyên lý

LightGBM-style baseline là model cây boosting trên feature tabular của từng dòng ngày-mã. Nó không dùng sequence 64 phiên trực tiếp, mà dựa vào feature engineering đã encode momentum, volatility, candle, volume, market-relative signal.

Trong môi trường hiện tại, `lightgbm` không có sẵn nên script dùng fallback:

```text
sklearn.ensemble.HistGradientBoostingRegressor
```

### 7.2. Parameter

Nếu `lightgbm` có sẵn:

```text
n_estimators = 500
learning_rate = 0.03
num_leaves = 31
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
n_jobs = -1
```

Fallback đang chạy:

```text
max_iter = 350
learning_rate = 0.04
max_leaf_nodes = 31
l2_regularization = 0.001
random_state = 42
```

### 7.3. Nhận xét

LightGBM-style HGBR là baseline nhanh và mạnh nhất trong nhóm non-deep theo `LongShort5`. Nó tận dụng feature relative/market-relative tốt, nhưng không học trực tiếp cấu trúc 64 phiên như neural models, và không có explicit direction/excess score blend như Hybrid xLSTM.

## 8. Kronos Zero-Shot

Implementation liên quan:

- `scripts/run_kronos_full_test.py`
- `scripts/compute_kronos_full_metrics.py`
- `src/vnstock/models/kronos/zero_shot.py`

### 8.1. Nguyên lý

Kronos là foundation model cho K-line, dùng zero-shot inference. Repo không fine-tune Kronos trong final suite.

Flow:

```text
historical OHLCV window [t-63, ..., t]
-> Kronos tokenizer
-> Kronos foundation model
-> predicted future close path
-> pred_ret_5d = pred_close[t+5] / close[t] - 1
```

Config chính:

```text
context_length = 64
horizon = 5
tokenizer = NeoQuasar/Kronos-Tokenizer-base
model = NeoQuasar/Kronos-small
device = CPU
```

### 8.2. Caveat

Metric Kronos hiện là reference chưa hoàn chỉnh:

```text
rows = 18,978
symbols = 61 / 95
dates = 327
```

Do đó không nên kết luận Kronos kém hơn toàn diện cho tới khi full 95-symbol run hoàn tất. Tuy nhiên snapshot hiện tại cho thấy zero-shot Kronos chưa tạo được stock-selection signal tốt trên data này:

```text
IC = 0.0069
RankIC = 0.0189
LongShort5 = -0.1264%
```

## 9. Training Objective

Các neural baseline trong final suite dùng multitask regression + direction:

```text
Loss = Huber(pred_ret_5d, true_ret_5d)
     + 0.1 * BCEWithLogits(direction_logit, true_dir_5d)
```

Parameter chính:

```text
epochs = 4
batch_size = 512
eval_batch_size = 1024
learning_rate = 5e-4
weight_decay = 1e-4
huber_delta = 0.05
direction_loss_weight = 0.1
patience = 2
checkpoint_metric = valid_loss
checkpoint_mode = min
```

Giải thích:

- Huber ổn hơn MSE cho return vì return có outlier và fat-tail.
- Direction BCE ép model học dấu tăng/giảm, thay vì direction chỉ là sản phẩm phụ của regression.
- Weight direction `0.1` giữ direction head có tác dụng nhưng không phá ranking signal.

## 10. Tại Sao Không Chọn Theo Direction Accuracy Thuần

Trong full universe, nhiều sample có return rất nhỏ quanh `0%`. Những sample này gần như nhiễu, nhưng direction accuracy vẫn tính đúng/sai ngang với các move lớn.

Ví dụ:

```text
true_ret_5d = +0.05% hoặc -0.05%
```

Về trading, hai case này gần như không khác biệt nhiều, nhưng metric accuracy vẫn phạt model nếu sai dấu. Vì vậy nếu chỉ chase raw `Direction_Acc`, model có thể học đoán class phổ biến và làm giảm IC/top-k.

Với stock-selection, câu hỏi đúng hơn là:

```text
Score có xếp được mã tốt lên top và mã xấu xuống bottom không?
```

Do đó quyết định production ưu tiên:

```text
IC, RankIC, Top5_Return, LongShort5, Top5_Direction_Acc
```

## 11. Kết Luận Kỹ Thuật

Model nên giữ làm production candidate hiện tại:

```text
Rank-Aware Calibrated Hybrid xLSTM
```

Lý do:

- Nó giữ `Hybrid xLSTM Direction-Excess Blend` làm source score chính.
- Nó thêm calibration được chọn trên validation theo objective top-5/long-short.
- Nó cải thiện `LongShort5`, cost-adjusted total return, Sharpe proxy và max drawdown trong backtest top-5.
- Trade-off: `IC` và `RankIC` thấp hơn baseline một chút, nên baseline Hybrid vẫn được giữ làm reference source score.

Model nên giữ làm baseline phụ:

- `Hybrid xLSTM Direction-Excess Blend`: source score ổn định, RankIC cao hơn bản calibrated.
- `LightGBM-style HGBR`: baseline tabular nhanh, mạnh, dễ debug.
- `CNN-LSTM`: neural baseline cạnh tranh, đáng giữ để kiểm tra robustness.

Model chưa nên promote:

- `TCN`: metric hiện yếu hơn rõ.
- `PatchTST`: metric hiện yếu hơn rõ.
- `Kronos`: chỉ là zero-shot reference, chưa full 95-symbol historical test.

## 12. Artifact Liên Quan

| Artifact | Path |
| --- | --- |
| Final metrics | `outputs/reports/final_top5_model_suite/top5_model_suite_metrics.csv` |
| Final report | `outputs/reports/final_top5_model_suite/top5_model_suite_report.md` |
| Final predictions | `outputs/final/model_suite_top5/` |
| Hybrid xLSTM prediction | `outputs/final/hybrid_xlstm_direction_excess_blend_predictions.parquet` |
| Rank-aware selected prediction | `outputs/final/rank_aware_calibrated_hybrid_predictions.parquet` |
| Final comparison figure | `outputs/figures/final_top5_model_suite/top5_model_suite_longshort.png` |
| Final suite runner | `scripts/run_final_top5_model_suite.py` |
| Rank-aware upgrade runner | `scripts/run_rank_aware_hybrid_upgrade.py` |
| Rank-aware upgrade report | `outputs/reports/rank_aware_hybrid_upgrade/rank_aware_upgrade_report.md` |
