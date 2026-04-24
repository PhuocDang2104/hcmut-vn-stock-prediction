# Data Map And Contract

Tài liệu này mô tả snapshot dữ liệu đang có thật trong repo tại thời điểm hiện tại, sau khi mở rộng train universe từ `85` mã Việt Nam cùng ngành với `core10` lên `95` mã bằng cách thêm `10` cổ phiếu nước ngoài dùng làm proxy công nghệ, AI và supply-chain cho Việt Nam.

## 1. Snapshot Hiện Tại

- Raw equity data phủ tới `2026-04-14`
- Universe train mặc định nằm ở `data/universes/vn_stock_universe_95_plus_foreign_tech.txt`
- Universe này gồm:
  - `85` mã Việt Nam cùng ngành với `core10`
  - `10` mã nước ngoài lấy qua `YF`
- Shared feature panel phủ tới `2026-04-07` vì target `5` phiên làm rơi `5` phiên cuối của mỗi mã
- Target chính hiện tại là `target_ret_5d`

Universe con vẫn được giữ lại:

- `data/universes/core10.txt`: subset benchmark gốc
- `data/universes/vn_stock_universe_85_same_sector_core10.txt`: universe VN-only
- `data/universes/foreign_tech_influence_10.txt`: 10 mã ngoại chỉ dùng để train

## 2. Bài Toán Thực Tế

Granularity hiện tại:

- 1 dòng = 1 mã x 1 phiên giao dịch
- Tần suất = daily
- Mọi model cùng xuất phát từ một shared panel chuẩn hóa
- Target chính = lợi nhuận 5 phiên tới (`target_ret_5d`)

Diễn giải ngắn:

- Tại ngày `t`, mô hình quan sát `open, high, low, close, volume` và các feature dẫn xuất
- Mô hình dự báo phần trăm thay đổi của giá đóng cửa tại `t + 5` phiên
- Prediction cuối cùng vẫn phải chuẩn hóa về prediction schema chung để so sánh model

## 3. Bản Đồ Dữ Liệu

| Tầng dữ liệu | Đường dẫn | Nội dung hiện có | Vai trò |
| --- | --- | --- | --- |
| Raw equity | `data/raw/vn_stock_daily/` | `95` CSV, tổng `191,132` dòng | OHLCV gốc cho train |
| Raw index | `data/raw/vn_index_daily/` | `VNINDEX.csv`, `VN30.csv`, tổng `4,129` dòng | Chỉ số tham chiếu, chưa join vào shared panel |
| Raw reference | `data/raw/reference/` | manifest, summary, calendar, VN exchange map, VN industry map, foreign metadata | Metadata để audit universe |
| Raw viz | `data/raw/_viz/` | chart và summary theo từng raw folder | Artifact EDA trực tiếp trên raw data |
| Interim | `data/interim/` | `merged_panel.parquet`, `cleaned_panel.parquet`, `quality_report.json` | Tầng staging sau ingest |
| Shared processed | `data/processed/shared/` | feature panel, market panel, split parquet, schema/meta | Canonical dataset cho mọi model |
| Model exports | `data/processed/xlstm_ts/`, `itransformer/`, `patchtst/`, `kronos/` | dữ liệu đã adapter theo từng model | Input trực tiếp cho model blocks |
| Outputs | `outputs/figures/`, `outputs/reports/`, `outputs/metrics/` | chart, summary, metric | Artifact đầu ra |

Ghi chú:

- Tên folder `data/raw/vn_stock_daily/` là legacy name; hiện folder này chứa cả mã Việt Nam lẫn `10` mã ngoại dùng cho train
- `market_panel.parquet` hiện vẫn giống `feature_panel.parquet`, chưa tách thành market-only panel riêng

## 4. Raw Equity Data

### 4.1. Source mix

Manifest raw hiện tại:

- `symbols_fetched = 95`
- `rows = 191132`
- `symbols_by_source = {"KBS": 85, "YF": 10}`

Raw rows theo source:

- `KBS`: `170,449` dòng
- `YF`: `20,683` dòng

### 4.2. 10 mã ngoại đang có trong train universe

Danh sách nằm ở `data/universes/foreign_tech_influence_10.txt`:

- `005930.KS`
- `2317.TW`
- `AAPL`
- `INTC`
- `NVDA`
- `QCOM`
- `AMD`
- `TSM`
- `GOOGL`
- `MSFT`

Metadata mô tả nhanh nằm ở:

- `data/raw/reference/foreign_symbol_metadata.csv`

Schema file này:

- `symbol`
- `company_name`
- `market`
- `country`
- `currency`
- `theme_bucket`
- `vietnam_link`
- `source`

### 4.3. Coverage raw hiện tại

Tổng quan:

- Số file: `95`
- Tổng dòng: `191,132`
- Date range toàn bộ raw equity: `2018-01-02` -> `2026-04-14`
- `value` hiện null `100%`
- Sau chuẩn hóa: không có duplicate `symbol,date`, không có thiếu `close`

Những mã có lịch sử ngắn nhất:

| Symbol | Rows | Date min | Date max | Source |
| --- | ---: | --- | --- | --- |
| SSB | 1,262 | 2021-03-24 | 2026-04-14 | KBS |
| OCB | 1,296 | 2021-01-28 | 2026-04-14 | KBS |
| MSB | 1,321 | 2020-12-23 | 2026-04-14 | KBS |
| HHV | 1,428 | 2018-01-23 | 2026-04-14 | KBS |
| AGG | 1,561 | 2020-01-09 | 2026-04-14 | KBS |
| SIP | 1,704 | 2019-06-13 | 2026-04-14 | KBS |
| PGV | 1,803 | 2018-03-21 | 2026-04-14 | KBS |
| VGI | 1,885 | 2018-09-25 | 2026-04-14 | KBS |
| TCB | 1,965 | 2018-06-04 | 2026-04-14 | KBS |
| VHM | 1,975 | 2018-05-17 | 2026-04-14 | KBS |
| FRT | 1,990 | 2018-04-26 | 2026-04-14 | KBS |
| TDM | 1,992 | 2018-01-02 | 2026-04-14 | KBS |
| TPB | 1,994 | 2018-04-19 | 2026-04-14 | KBS |
| SGT | 2,000 | 2018-01-02 | 2026-04-13 | KBS |
| 2317.TW | 2,006 | 2018-01-02 | 2026-04-14 | YF |

Những mã dài nhất hiện tại:

- `AAPL, AMD, GOOGL, INTC, MSFT, NVDA, QCOM, TSM`: `2,081` dòng mỗi mã

### 4.4. Raw schema

Mọi CSV raw equity dùng cùng schema:

| Column | Kiểu hiện tại | Null | Ý nghĩa |
| --- | --- | ---: | --- |
| `symbol` | string | 0 | Mã cổ phiếu |
| `date` | string trong CSV raw | 0 | `YYYY-MM-DD` |
| `open` | float64 | 0 | Giá mở cửa |
| `high` | float64 | 0 | Giá cao nhất |
| `low` | float64 | 0 | Giá thấp nhất |
| `close` | float64 | 0 | Giá đóng cửa |
| `volume` | int64/float castable | 0 | Khối lượng |
| `value` | float64 | 191,132 | Hiện đang null toàn bộ |
| `source` | string | 0 | `KBS` hoặc `YF` |

## 5. Raw Index Và Reference Data

### 5.1. Raw index

Hiện có:

| Symbol | Rows | Date min | Date max |
| --- | ---: | --- | --- |
| VNINDEX | 2,065 | 2018-01-02 | 2026-04-14 |
| VN30 | 2,064 | 2018-01-02 | 2026-04-14 |

Schema của index giống raw equity.

### 5.2. Reference files

`data/raw/reference/` hiện có:

| File | Nội dung | Ghi chú |
| --- | --- | --- |
| `raw_data_manifest.json` | snapshot fetch config và universe | manifest chính |
| `raw_symbol_summary.csv` | coverage theo mã | gồm cả `KBS` và `YF` |
| `trading_calendar.csv` | lịch phiên từ toàn bộ raw equity | nay là union calendar của nhiều thị trường |
| `exchange_mapping.csv` | exchange map cho phần VN universe | hiện chỉ cover `85` mã Việt Nam |
| `industry_mapping.csv` | ngành cho phần VN universe | lấy từ fallback `KBS` |
| `foreign_symbol_metadata.csv` | metadata 10 mã ngoại | riêng cho foreign train proxies |
| `reference_fetch_warnings.json` | log fallback reference | hiện ghi lại lỗi `VCI` industry endpoint |
| `raw_download_errors.json` | lỗi raw fetch | snapshot hiện tại là `{"errors": []}` |

### 5.3. Metadata theo thị trường

VN exchange map hiện có:

- `HSX`: `84` mã
- `UPCOM`: `1` mã (`VGI`)

Foreign metadata hiện có theo market:

- `NASDAQ`: `7`
- `KRX`: `1`
- `TWSE`: `1`
- `NYSE`: `1`

### 5.4. Trading calendar

`trading_calendar.csv` hiện có:

- `2,154` phiên
- `max_gap = 4`

Quan trọng:

- Từ khi thêm Mỹ, Hàn, Đài Loan, calendar này không còn là lịch giao dịch Việt Nam thuần nữa
- Đây là union của toàn bộ ngày xuất hiện trong raw panel mixed-market

## 6. Interim Layer

`data/interim/` hiện có:

| File | Rows | Vai trò |
| --- | ---: | --- |
| `merged_panel.parquet` | 191,132 | bản gộp từ raw CSV |
| `cleaned_panel.parquet` | 191,132 | bản chuẩn hóa dùng để build shared dataset |
| `quality_report.json` | n/a | summary chất lượng dữ liệu |

`quality_report.json` hiện báo:

- `symbols = 95`
- `rows = 191132`
- `date_min = 2018-01-02`
- `date_max = 2026-04-14`
- `missing_close = 0`
- `duplicates = 0`

## 7. Shared Feature Panel

### 7.1. Shared panel là gì

`data/processed/shared/feature_panel.parquet` là canonical dataset dùng chung cho mọi model.

Snapshot hiện tại:

- Rows: `190,657`
- Date range: `2018-01-02` -> `2026-04-07`
- Split counts:
  - `train = 137,377`
  - `valid = 23,745`
  - `test = 29,535`

Chênh lệch raw vs feature panel:

- Raw equity có `191,132` dòng
- Feature panel có `190,657` dòng
- Chênh lệch `475` dòng = `95 mã x 5 phiên cuối`

### 7.2. Feature engineering đúng theo code hiện tại

Các feature đang bật:

- `ret_1d`
- `ret_5d`
- `log_volume`
- `hl_spread`
- `oc_change`
- `rolling_vol_5`
- `rolling_vol_20`
- `ma_5`
- `ma_20`
- `ma_ratio_5_20`

Target và cột bổ sung:

- `target_ret_1d`
- `target_ret_5d`
- `target_dir_5d`
- `time_idx`
- `group_id`

### 7.3. Shared schema

Schema canonical hiện tại:

- `symbol`
- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `value`
- `ret_1d`
- `ret_5d`
- `log_volume`
- `hl_spread`
- `oc_change`
- `rolling_vol_5`
- `rolling_vol_20`
- `ma_5`
- `ma_20`
- `ma_ratio_5_20`
- `target_ret_1d`
- `target_ret_5d`
- `split`
- `time_idx`
- `group_id`
- `target_dir_5d`

Null status hiện tại:

- mọi feature và target trong shared panel đều đã được fill/drop để không còn null
- `value` vẫn null toàn bộ vì carry từ raw

### 7.4. Target distribution

Toàn bộ panel:

- mean = `0.004300`
- std = `0.057959`
- min = `-0.382004`
- median = `0.001768`
- max = `1.001125`
- positive rate = `50.81%`

Theo split:

| Split | Rows | Date min | Date max | Positive rate |
| --- | ---: | --- | --- | ---: |
| train | 137,377 | 2018-01-02 | 2023-12-29 | 0.5075 |
| valid | 23,745 | 2024-01-02 | 2024-12-31 | 0.4989 |
| test | 29,535 | 2025-01-02 | 2026-04-07 | 0.5183 |

## 8. Split Logic

Boundary config:

- `train_end = 2023-12-31`
- `valid_end = 2024-12-31`
- `test_end = 2026-12-31`

Những mã có ít dòng nhất theo split:

Train:

- `SSB = 696`
- `OCB = 730`
- `MSB = 755`
- `HHV = 862`
- `AGG = 995`

Valid:

- `2317.TW = 242`
- `TDM = 243`
- `005930.KS = 244`
- phần lớn các mã khác = `250`

Test:

- `TDM = 289`
- `2317.TW = 300`
- `005930.KS = 305`
- `SGT = 309`
- `PGV = 310`
- phần lớn các mã khác = `311`

Giải thích:

- split không còn đồng đều theo symbol vì khác nhau về lịch sử niêm yết và khác nhau về market calendar
- điều này đặc biệt rõ ở `KRX` và `TWSE`

## 9. Model-Specific Exports

### 9.1. xLSTM-TS

Đường dẫn:

- `data/processed/xlstm_ts/`

Snapshot hiện tại:

| Split | `X` shape | `y` shape |
| --- | --- | --- |
| train | `[131392, 64, 10]` | `[131392]` |
| valid | `[23745, 64, 10]` | `[23745]` |
| test | `[29535, 64, 10]` | `[29535]` |

### 9.2. iTransformer

Đường dẫn:

- `data/processed/itransformer/`

Format:

- `train.parquet`
- `valid.parquet`
- `test.parquet`
- `meta.json` với `seq_len = 64`, `pred_len = 5`

### 9.3. PatchTST

Đường dẫn:

- `data/processed/patchtst/`

Format:

- parquet split
- `meta.json` với `seq_len = 64`, `pred_len = 5`, `patch_len = 16`, `patch_stride = 8`

### 9.4. Kronos

Đường dẫn:

- `data/processed/kronos/`

Format:

- CSV split
- có thêm `timestamp`
- có thêm `target`
- `meta.json` ghi `adapter_mode = vn_equity_daily`

## 10. Visual Artifacts Đã Cập Nhật

Raw EDA tổng quát:

- `outputs/figures/raw_data/raw_universe_rebased_close.png`
- `outputs/figures/raw_data/raw_universe_volume_boxplot.png`
- `outputs/figures/raw_data/raw_universe_return_correlation.png`
- `outputs/figures/raw_data/raw_universe_monthly_coverage.png`
- `outputs/reports/raw_data_summary.md`

Folder-level viz:

- `data/raw/_viz/vn_stock_daily/`
- `data/raw/_viz/vn_index_daily/`
- `data/raw/_viz/reference/`

Reference viz hiện có thêm:

- `foreign_symbols_by_market.png`

Repo cũng sync các artifact tương ứng sang:

- `outputs/figures/folder_viz/`
- `outputs/reports/folder_viz/`

## 11. Prediction Output Contract

Prediction export để so sánh model phải dùng schema:

- `model_family`
- `model_version`
- `symbol`
- `date`
- `split`
- `y_true`
- `y_pred`
- `target_name`
- `horizon`
- `run_id`

## 12. Caveat Quan Trọng

- `value` hiện null `100%` ở raw, interim và shared
- `exchange_mapping.csv` và `industry_mapping.csv` chỉ cover phần VN universe; 10 mã ngoại nằm riêng trong `foreign_symbol_metadata.csv`
- `trading_calendar.csv` giờ là mixed-market union calendar, không còn là lịch Việt Nam thuần
- shared panel hiện không carry `source`, `market`, `country`, `currency`; model chỉ thấy chuỗi feature theo symbol
- feature `ma_5` và `ma_20` là giá tuyệt đối, nên khi trộn `VND`, `USD`, `KRW`, `TWD` sẽ có khác biệt scale đáng kể
- các model dạng array như `xLSTM-TS` hiện cũng không có static market tag, nên mixed-source training cần được đọc với caveat trên
- `VNINDEX` và `VN30` vẫn chưa được join vào feature panel
- fill feature hiện vẫn dùng `ffill().bfill()` theo từng symbol, nên đầu chuỗi có khả năng leakage nhẹ
- dữ liệu train/eval luôn kết thúc sớm hơn raw đúng `5` phiên cho mỗi mã vì target horizon
