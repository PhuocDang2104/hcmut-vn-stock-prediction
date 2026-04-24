# **ĐỀ XUẤT KỸ THUẬT ÁP DỤNG xLSTM-TS CHO DỮ LIỆU KLINE ĐA MÃ, ĐA BIẾN, HORIZON 5 PHIÊN**

## **1\. Mục tiêu**

Tài liệu này tóm tắt cách áp dụng xLSTM-TS vào bộ dữ liệu kline daily hiện tại gồm 95 mã, với mục tiêu dự báo `target_ret_5d` một cách ổn định, đúng bản chất dữ liệu, và đủ chặt để dùng cho cả train, đánh giá, và inference. Nền tảng tham chiếu gồm xLSTM gốc, biến thể xLSTMTime cho forecasting, và implementation hiện có trong PyTorch Forecasting.

## **2\. xLSTM-TS là gì**

xLSTM là họ mô hình mở rộng từ LSTM cổ điển, thay cơ chế gating truyền thống bằng exponential gating và bổ sung memory structure mới nhằm tăng khả năng lưu giữ phụ thuộc dài hạn. Paper gốc giới thiệu hai biến thể chính là `sLSTM` và `mLSTM`, sau đó xLSTMTime điều chỉnh backbone này cho bài toán dự báo chuỗi thời gian nhiều biến.

### **2.1. Link paper tham chiếu**

* xLSTM: Extended Long Short-Term Memory  
   [`https://arxiv.org/abs/2405.04517`](https://arxiv.org/abs/2405.04517)   
* xLSTMTime: Long-term Time Series Forecasting With xLSTM  
   [`https://arxiv.org/abs/2407.10240`](https://arxiv.org/abs/2407.10240) 

### **2.2. Khi nào xLSTM-TS phù hợp**

xLSTM-TS phù hợp khi bài toán có đặc điểm: chuỗi theo thời gian rõ ràng, cần học phụ thuộc dài hơn LSTM chuẩn, dữ liệu có nhiều feature động trên mỗi time step, và cần một recurrent backbone hiện đại nhưng vẫn giữ inductive bias theo thời gian. Với kline daily đa mã, đây là một lựa chọn hợp lý hơn so với việc ép toàn bộ bài toán về panel đồng bộ kiểu transformer-only.

## **3\. Nguyên lý thuật toán**

### **3.1. Ý tưởng cốt lõi**

LSTM cổ điển mạnh ở việc truyền trạng thái qua thời gian, nhưng thường yếu dần khi cần scale lớn hoặc học phụ thuộc dài hơn. xLSTM cải tiến điểm này bằng:

* exponential gating để điều tiết dòng thông tin mạnh hơn  
* memory structure mới giúp tăng sức chứa biểu diễn  
* residual block stacking để huấn luyện sâu hơn và ổn định hơn

### **3.2. Hai biến thể chính**

**sLSTM**

* dùng scalar memory  
* thiên về ổn định huấn luyện  
* phù hợp để làm baseline đầu tiên cho dữ liệu tài chính daily

**mLSTM**

* dùng matrix memory  
* giàu năng lực biểu diễn hơn  
* phù hợp để mở rộng sau khi pipeline dữ liệu và evaluation đã ổn định

### **3.3. Dạng bài toán trong repo hiện tại**

Với dữ liệu của bạn, bài toán nên được hiểu là sequence-to-one regression trên từng symbol. Mỗi sample là một cửa sổ lịch sử độ dài `L`, đầu vào có `F` feature tại mỗi phiên, và đầu ra là lợi nhuận 5 phiên tới:

Xi,t​∈RLF

yi,t​≈ yi,t​=target\_ret\_5d(i,t)

Cách hiểu này đúng hơn nhiều so với việc xem toàn bộ 95 mã là 95 channel đồng bộ tuyệt đối trên cùng một calendar.

## **4\. Dữ liệu hiện có và cách fit với model**

## **4.1. Dữ liệu đang có**

Hiện tại bạn có:

* raw equity cho 95 mã  
* shared feature panel là canonical dataset cho mọi model  
* target chính là `target_ret_5d`  
* split theo thời gian thành train, valid, test  
* horizon thực tế là 5 phiên  
* mixed-market universe gồm Việt Nam, Mỹ, Hàn Quốc, Đài Loan

Điểm cần lưu ý là shared panel hiện chưa carry đầy đủ market context như `market`, `currency`, `country`, `source`; trong khi calendar hiện là union calendar của nhiều thị trường. Vì vậy, nếu train trực tiếp bằng giá tuyệt đối thì model rất dễ học sai tín hiệu do chênh lệch scale và regime giao dịch.

## **4.2. Cách tổ chức sample đúng**

Mỗi sample nên là một chuỗi lịch sử của một symbol:

* chiều thời gian: `seq_len = 64`  
* chiều feature: `F` feature đã chuẩn hóa  
* nhãn: `target_ret_5d` tại thời điểm cuối cửa sổ

Biểu diễn:

Xi,t​=\[xi,t−L+1​,…,xi,t​\]  
yi,t=Closei,t+5​− Closei,t​Closei,t

hoặc dùng trực tiếp giá trị `target_ret_5d` đã có trong shared panel. Cách này khớp tự nhiên với recurrent forecasting.

## **5\. Biến đổi và chuẩn hóa dữ liệu để fit với xLSTM-TS**

## **5.1. Nguyên tắc chuẩn hóa**

Nếu train chung cả 95 mã, không nên chuẩn hóa foreign theo kiểu đổi toàn bộ về VND rồi giữ giá tuyệt đối. Cách đúng là đưa tất cả mã về cùng “ngôn ngữ thống kê” bằng feature tương đối và scaling theo từng symbol. Điều model cần học là động học giá, không phải mệnh giá tiền tệ.

## **5.2. Feature nên giữ**

Nên ưu tiên các feature đã tương đối hoặc gần tương đối:

* `ret_1d`  
* `ret_5d`  
* `log_volume`  
* `hl_spread`  
* `oc_change`  
* `rolling_vol_5`  
* `rolling_vol_20`  
* `ma_ratio_5_20`

Nhóm này phù hợp với mixed-market training hơn vì ít phụ thuộc đơn vị tiền tệ.

## **5.3. Feature giá tuyệt đối cần biến đổi**

Các biến `open`, `high`, `low`, `close`, `ma_5`, `ma_20` không nên dùng trực tiếp khi train chung 95 mã. Nên chuyển sang dạng tương đối:

open\_relt​=opentcloset​​−1  
high\_relt​=hightcloset​​−1  
low\_relt​=lowtcloset​​−1  
ma5\_relt​=ma5(t)closet​​−1  
ma20\_relt​=ma5(t)closet​​−1

Cách biến đổi này loại bỏ lệch scale giữa VND, USD, KRW và TWD nhưng vẫn giữ được hình dạng động học của chuỗi.

## **5.4. Scaling phù hợp**

Khuyến nghị fit scaler theo từng symbol trên train split:

zi,t(f)=xi,t(f)- μi(f)σi(f)+ ϵ​​

Trong đó μi(f) và σi(f) chỉ được ước lượng từ train split của symbol `i`. Với volume hoặc volatility có tail dày, có thể dùng robust scaling thay cho z-score. Không nên dùng global scaler cho toàn bộ 95 mã.

## **5.5. Market context**

Nếu có thể mở rộng dataset adapter, nên bổ sung static metadata:

* `market`  
* `currency`  
* `country`  
* `source`

Các biến này không thay thế normalization, nhưng giúp model hiểu rằng một chuỗi đến từ Việt Nam hay Mỹ, từ đó học regime tốt hơn. Đây là phần rất đáng làm nếu bạn muốn giữ 10 mã foreign làm proxy trong train universe.

## **6\. Kiến trúc đề xuất và ý nghĩa từng phần**

## **6.1. Luồng tổng thể**

Kiến trúc nên được hiểu theo chuỗi xử lý sau:

1. Input sequence  
2. Normalization hoặc scaling đã fit từ train  
3. Series decomposition nếu dùng  
4. xLSTM blocks  
5. Projection head  
6. Regression output

Documentation của PyTorch Forecasting mô tả xLSTMTime như một forecasting model kết hợp normalization layer, decomposition layer và backbone xLSTM để tạo forecast ổn định hơn trên chuỗi dài.

## **6.2. Input layer**

Input layer nhận tensor:

X∈RB×L×F

Trong đó:

* `B`: batch size  
* `L`: sequence length  
* `F`: số feature

Ý nghĩa của lớp này là đưa chuỗi lịch sử nhiều biến của từng symbol vào backbone recurrent. Với bài toán hiện tại, `L = 64` là điểm khởi đầu hợp lý.

## **6.3. Normalization layer**

Lớp này giúp ổn định phân phối đầu vào và giảm ảnh hưởng của outlier hoặc scale chênh lệch giữa feature. Trong bài toán của bạn, phần quan trọng hơn cả kiến trúc là normalization đúng cách trước khi vào model. Nếu normalization sai, xLSTM mạnh đến đâu cũng dễ học theo scale thay vì pattern.

## **6.4. Decomposition layer**

xLSTMTime hỗ trợ decomposition kernel để tách thành phần xu hướng và dao động của chuỗi. Ý nghĩa của lớp này là giúp model học phần trend và phần residual riêng rẽ. Với dữ liệu tài chính daily và horizon 5 phiên, decomposition có thể hữu ích nhưng không nên mặc định coi là luôn tốt hơn; cần kiểm chứng bằng ablation.

## **6.5. xLSTM blocks**

Đây là phần lõi của mô hình. Mỗi block gồm cơ chế memory update, gating và biến đổi trạng thái ẩn. Nếu dùng `sLSTM`, block ưu tiên ổn định và phù hợp cho baseline. Nếu dùng `mLSTM`, block có khả năng biểu diễn tương tác giàu hơn nhưng nhạy hơn với cấu hình. Backbone này là phần trực tiếp học phụ thuộc ngắn và dài trong chuỗi kline.

## **6.6. Projection head**

Projection head nhận hidden state cuối hoặc representation tổng hợp của chuỗi, sau đó biến đổi sang forecast output. Với bài toán hiện tại, output chính là một scalar cho mỗi sample. Ý nghĩa của head này là ánh xạ representation thời gian sang dự báo return.

## **7\. Thiết kế output và objective**

## **7.1. Output khuyến nghị**

Output nên là một scalar regression:

yi,t​ ∈ R 

trong đó:

yi,t​ ≈ yi,t​=target\_ret\_5d(i,t)  

Đây là thiết kế gọn, đúng contract hiện tại, và dễ so sánh với các model khác.

## **7.2. Objective**

Khuyến nghị dùng Huber loss cho regression:

![][image1]

Lý do là return tài chính thường có outlier và tail dày, nên Huber thường ổn hơn MSE thuần. Có thể benchmark thêm với MAE hoặc MSE, nhưng Huber nên là objective mặc định ban đầu.

## **7.3. Prediction schema**

Đầu ra phục vụ so sánh model nên bám đúng contract:

* `model_family`  
* `model_version`  
* `symbol`  
* `date`  
* `split`  
* `y_true`  
* `y_pred`  
* `target_name`  
* `horizon`  
* `run_id`

Trong đó:

* `target_name = target_ret_5d`  
* `horizon = 5`

## **8\. Đánh giá mô hình**

## **8.1. Không chỉ nhìn loss**

Với forecasting cổ phiếu, chỉ nhìn loss là chưa đủ. Nên đánh giá theo ba nhóm.

**Nhóm 1: sai số hồi quy**

* Huber  
* MAE  
* RMSE

**Nhóm 2: chất lượng xếp hạng**

* Pearson IC  
* Spearman RankIC

**Nhóm 3: chất lượng định hướng và lựa chọn**

* directional accuracy  
* top-k realized return  
* spread giữa nhóm dự báo cao và thấp

Cách đánh giá này sát hơn với use case stock selection so với chỉ tối ưu sai số số học.

## **8.2. Validation strategy**

Nên giữ split theo thời gian như hiện tại, không shuffle theo ngày giữa train, valid, test. Early stopping nên theo một metric tổng hợp, ưu tiên valid Huber loss kết hợp valid IC. Điều này giúp tránh chọn model chỉ khớp biên độ mà không có giá trị xếp hạng.

## **9\. Thiết kế inference**

## **9.1. Inference offline trên test**

Cho mỗi symbol:

* lấy `seq_len` phiên gần nhất có đủ feature  
* áp scaler đúng version đã fit từ train  
* đưa qua model  
* nhận `y_pred`  
* xuất đúng prediction contract

## **9.2. Inference production-like**

Tại ngày `t`:

* cập nhật raw data  
* build lại feature cho ngày `t`  
* lấy cửa sổ `t-L+1` đến `t`  
* transform bằng scaler train-time  
* chạy model để thu được dự báo cho horizon 5 phiên

Điểm bắt buộc là feature order, scaler version, seq\_len và target definition phải giống hệt lúc train. Nếu lệch một trong các phần này, inference sẽ sai contract dù code vẫn chạy.

## **9.3. Gói artifact nên lưu**

Nên lưu cùng model:

* checkpoint trọng số  
* `model_config.json`  
* `feature_list.json`  
* scaler stats  
* target definition  
* training metadata  
* run id

## **10\. Caveat quan trọng**

## **10.1. Mixed-currency**

Đây là rủi ro lớn nhất nếu dùng giá tuyệt đối. Foreign không nên bị ép về VND rồi train bằng price level. Cách đúng là dùng feature tương đối cộng với per-symbol normalization.

## **10.2. Mixed-calendar**

Calendar hiện là union của nhiều thị trường. Không nên ép toàn bộ mã vào một lưới ngày giả đồng bộ rồi fill mạnh tay. Hợp lý hơn là build sequence riêng cho từng symbol.

## **10.3. Leakage do backfill**

Nếu feature engineering có `bfill` đầu chuỗi thì recurrent model có thể hưởng lợi giả tạo từ thông tin tương lai. Khuyến nghị bỏ `bfill` cho feature động và chỉ cho phép sample sau khi cửa sổ rolling đã hình thành tự nhiên.

## **10.4. Thiếu market context**

Nếu shared panel chưa carry `market`, `currency`, `country`, `source`, thì model khó phân biệt regime giữa Việt Nam và foreign. Đây là lý do nên thêm static metadata nếu muốn train chung cả 95 mã một cách sạch hơn.

## **11\. Cấu hình khuyến nghị để bắt đầu**

### **Dữ liệu**

* dùng sequence theo từng symbol  
* `seq_len = 64`  
* bỏ `value`  
* bỏ giá tuyệt đối nếu chưa chuyển sang feature tương đối  
* scale theo từng symbol trên train split

### **Feature đầu vào**

* `ret_1d`  
* `ret_5d`  
* `log_volume`  
* `hl_spread`  
* `oc_change`  
* `rolling_vol_5`  
* `rolling_vol_20`  
* `ma_ratio_5_20`  
* `open_rel`  
* `high_rel`  
* `low_rel`  
* `ma5_rel`  
* `ma20_rel`

### **Model**

* `xlstm_type = slstm`  
* `hidden_size = 128` hoặc `192`  
* `num_layers = 2` hoặc `3`  
* `dropout = 0.1`

### **Loss**

* Huber

### **Theo dõi**

* valid loss  
* valid IC  
* test IC  
* top-k realized return

Cấu hình này đủ gọn để benchmark nhưng vẫn bám đúng bản chất dữ liệu hiện tại.

## **12\. Kết luận**

xLSTM-TS là lựa chọn phù hợp cho bài toán kline daily đa mã nếu dữ liệu được tổ chức đúng: mỗi sample là chuỗi lịch sử của một symbol, input ưu tiên feature tương đối, normalization làm theo từng symbol, và có market context nếu train chung cả 95 mã. Nếu giữ nguyên giá tuyệt đối mixed-currency và không bổ sung metadata thị trường, model vẫn train được nhưng độ chính xác khó bền và rất dễ học sai tín hiệu. Nút thắt chính không nằm ở việc chọn `sLSTM` hay `mLSTM`, mà nằm ở cách chuẩn hóa feature, thiết kế sequence, kiểm soát leakage, và định nghĩa evaluation cho đúng mục tiêu đầu tư.

## **13\. Tài liệu tham chiếu**

* xLSTM: Extended Long Short-Term Memory  
   `https://arxiv.org/abs/2405.04517`  
* xLSTMTime: Long-term Time Series Forecasting With xLSTM  
   `https://arxiv.org/abs/2407.10240`  
* xLSTMTime documentation trong PyTorch Forecasting  
   `https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.xlstm._xlstm.xLSTMTime.html`

[image1]: <data:image/png;base64,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>