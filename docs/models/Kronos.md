# **ĐỀ XUẤT KỸ THUẬT ÁP DỤNG Kronos CHO DỮ LIỆU KLINE ĐA MÃ, ĐA BIẾN, HORIZON 5 PHIÊN**

## **1\. Mục tiêu**

Tài liệu này tóm tắt cách áp dụng Kronos vào bộ dữ liệu kline daily hiện tại gồm 95 mã, với mục tiêu dự báo `target_ret_5d` một cách ổn định, đúng bản chất dữ liệu, và đủ chặt để dùng cho cả train, đánh giá, và inference. Nền tảng tham chiếu gồm paper Kronos gốc, implementation chính thức, và data contract hiện có trong repo. Trong bài toán này, Kronos không nên được dùng như một model forecasting generic ăn feature panel tùy ý, mà nên được giữ gần với backbone nguyên bản: đầu vào downstream vẫn mang semantics K-line, nhưng đã được chuẩn hóa theo hướng phù hợp với multi-market setting.

## **2\. Kronos là gì**

Kronos là một foundation model dành riêng cho “ngôn ngữ” của thị trường tài chính, cụ thể là chuỗi K-line/candlestick. Thay vì dự báo trực tiếp trên không gian liên tục bằng một backbone forecasting tiêu chuẩn, Kronos đi theo hướng **discretize-and-generate**: trước hết mô hình dùng một tokenizer để ánh xạ dữ liệu thị trường liên tục sang token rời rạc, sau đó một decoder-only Transformer học phân phối của chuỗi token theo thời gian bằng objective tự hồi quy. Cách tiếp cận này giúp Kronos tận dụng tri thức pretraining quy mô lớn trên dữ liệu tài chính và phù hợp hơn với đặc điểm nhiễu cao, tail dày, và regime shift thường gặp trong thị trường.

### **2.1. Link paper tham chiếu**

Kronos: *A Foundation Model for the Language of Financial Markets*  
arXiv: `https://arxiv.org/abs/2508.02739`

Official implementation  
GitHub: `https://github.com/Leegoo-dev/Kronos`

Model card / usage mirrors  
Hugging Face: Kronos-base, Kronos-small

### **2.2. Khi nào Kronos phù hợp**

Kronos phù hợp khi bài toán có đặc điểm:

* chuỗi theo thời gian rõ ràng  
* đầu vào mang bản chất K-line hoặc gần với K-line  
* muốn tận dụng pretraining trên dữ liệu tài chính quy mô lớn  
* downstream task vẫn đủ gần miền OHLCV để tokenizer và codebook phát huy tác dụng  
* không bắt buộc mọi model trong benchmark phải dùng đúng cùng một generic feature schema

Với kline daily đa mã, Kronos đặc biệt phù hợp nếu đầu vào downstream vẫn được tổ chức như một chuỗi OHLCV đã chuẩn hóa theo hướng relative/return. Nếu ép mô hình sang một panel feature-engineering quá xa khỏi K-line semantics, lợi thế lớn nhất của Kronos sẽ suy giảm đáng kể.

## **3\. Nguyên lý thuật toán**

### **3.1. Ý tưởng cốt lõi**

Ý tưởng cốt lõi của Kronos là xem dữ liệu K-line như một ngôn ngữ có thể được token hóa. Mỗi quan sát thị trường liên tục tại một thời điểm không được đưa thẳng vào một regressor trong không gian thực, mà được biến đổi thành token rời rạc để mô hình học các mẫu price dynamics và trade activity patterns trong không gian token. Nhờ đó, Kronos có thể tận dụng khung pretraining tương tự LLM nhưng trên dữ liệu tài chính, thay vì chỉ học một mô hình mapping ngắn hạn giữa tensor đầu vào và scalar đầu ra.

### **3.2. Hai thành phần chính**

Kronos gồm hai phần chính:

**Tokenizer K-line**

Tokenizer chuyên biệt rời rạc hóa dữ liệu thị trường liên tục thành token sequence, đồng thời bảo toàn cả động học giá và mẫu hoạt động giao dịch. Paper mô tả tokenizer theo hướng phân cấp coarse-to-fine để biểu diễn market record ở nhiều mức chi tiết. Đây là nơi codebook được học và là lý do distribution alignment trở thành vấn đề quan trọng ở downstream.

**Autoregressive predictor**

Predictor là một decoder-only Transformer học chuỗi token bằng objective tự hồi quy. Phần này chịu trách nhiệm mô hình hóa phụ thuộc thời gian và sinh token tương lai từ lịch sử token đã có. Đây là backbone chính dùng cho forecasting, volatility forecasting và cả generative tasks trong paper.

### **3.3. Dạng bài toán trong repo hiện tại**

Với dữ liệu của bạn, bài toán nên được hiểu là sequence-to-one regression trên từng symbol. Mỗi sample là một cửa sổ lịch sử độ dài **L**, đầu vào có **F** biến K-line đã chuẩn hóa tại mỗi phiên, và đầu ra là lợi nhuận 5 phiên tới:

Xi,t​∈RLF

yi,t​≈ yi,t​=target\_ret\_5d(i,t)

Cách hiểu này đúng hơn nhiều so với việc xem toàn bộ 95 mã là 95 channel đồng bộ tuyệt đối trên cùng một calendar. Trong repo hiện tại, export của Kronos đã được adapter thành CSV split, có thêm `timestamp`, `target`, và `meta.json` ghi `adapter_mode = vn_equity_daily`, nên hướng triển khai hợp lý là giữ riêng adapter cho Kronos thay vì ép nó về cùng schema với iTransformer hay xLSTM-TS.

## **4\. Dữ liệu hiện có và cách fit với model**

### **4.1. Dữ liệu đang có**

Hiện tại bạn có:

* raw equity cho 95 mã  
* shared feature panel là canonical dataset cho mọi model  
* target chính là `target_ret_5d`  
* split theo thời gian thành train, valid, test  
* horizon thực tế là 5 phiên  
* mixed-market universe gồm Việt Nam, Mỹ, Hàn Quốc, Đài Loan

Điểm cần lưu ý là shared panel hiện chưa carry đầy đủ market context như `market`, `currency`, `country`, `source`; trong khi calendar hiện là union calendar của nhiều thị trường. Vì vậy, nếu train trực tiếp bằng giá tuyệt đối thì model rất dễ học sai tín hiệu do chênh lệch scale và regime giao dịch. Với Kronos, rủi ro này còn nhạy hơn vì tokenizer/codebook cũng có thể bị ảnh hưởng nếu phân phối downstream lệch mạnh khỏi miền kỳ vọng.

### **4.2. Cách tổ chức sample đúng**

Mỗi sample nên là một chuỗi lịch sử của một symbol:

* chiều thời gian: `seq_len = 64`  
* chiều feature: F biến OHLCV đã chuẩn hóa theo đúng schema Kronos downstream  
* nhãn: `target_ret_5d` tại thời điểm cuối cửa sổ

Biểu diễn:

Xi,t​=\[xi,t−L+1​,…,xi,t​\]  
yi,t=Closei,t+5​− Closei,t​Closei,t

hoặc dùng trực tiếp giá trị `target_ret_5d` đã có trong shared panel.

Điểm quan trọng là F ở đây không nên là một arbitrary feature panel. Với Kronos, F nên giữ semantics K-line càng nhiều càng tốt, tức OHLCV đã chuẩn hóa theo hướng relative/return, có thể thêm `amount/value` nếu dữ liệu thật sự sạch.

## **5\. Biến đổi và chuẩn hóa dữ liệu để fit với Kronos**

### **5.1. Nguyên tắc chuẩn hóa**

Nếu train chung cả 95 mã, không nên đổi toàn bộ về VND rồi giữ giá tuyệt đối. Cách đúng là đưa tất cả mã về cùng ngôn ngữ thống kê bằng các biến relative/return, nhưng vẫn giữ semantics của K-line. Điều model cần học là động học giá, không phải mệnh giá tiền tệ. Tuy nhiên, khác với một số backbone forecasting khác, Kronos không nên bị normalize quá mạnh theo kiểu z-score per-symbol ngay từ đầu, vì tokenizer/codebook đã được học trên một phân phối tài chính toàn cục; kéo từng mã về một phân phối gần N(0,1) có thể làm lệch token distribution downstream.

### **5.2. Feature nên giữ**

Nếu triển khai Kronos theo hướng hợp lý nhất trong bài toán này, nên ưu tiên các biến gần với OHLCV gốc:

* `open_rel`  
* `high_rel`  
* `low_rel`  
* `close_ret` hoặc `log_return`  
* `volume_log` hoặc `volume_robust`  
* `amount` hoặc `value` chỉ khi dữ liệu sạch và nhất quán

Tức là vẫn là “OHLCV”, nhưng ở dạng đã chuẩn hóa thống kê, không phải raw absolute level. Đây là điểm khác biệt quan trọng so với cách build feature panel generic cho các model khác.

### **5.3. Feature giá tuyệt đối cần biến đổi**

Các biến open, high, low, close không nên dùng trực tiếp ở dạng absolute price khi train multi-market. Nên chuyển sang dạng tương đối:

open\_relt​=opentcloset​​−1  
high\_relt​=hightcloset​​−1  
low\_relt​=lowtcloset​​−1  
ma5\_relt​=ma5(t)closet​​−1  
ma20\_relt​=ma5(t)closet​​−1

hoặc:

rt​=log(closetcloset-1) 

Cách biến đổi này loại bỏ phần lớn lệch scale giữa VND, USD, KRW và TWD nhưng vẫn giữ được hình dạng động học của chuỗi K-line.

### **5.4. Scaling phù hợp**

Với volume hoặc amount, có thể dùng log transform hoặc robust scaling nhẹ:

volume\_logt​=log(1+volumet​) 

Nếu thật sự cần thêm normalization, nên ưu tiên:

* global stats cross-symbol  
* hoặc per-market stats như VN, US, KR, TW

Không nên lấy z-score per-symbol làm mặc định cho toàn bộ đầu vào, vì cách đó có thể phá alignment giữa downstream distribution và codebook của tokenizer. Với Kronos, distribution alignment quan trọng không kém feature semantics.

### **5.5. Market context**

Nếu có thể mở rộng dataset adapter, nên bổ sung static metadata hoặc context token như:

* `market_id`  
* `currency`  
* `country`  
* `source`  
* `timezone`  
* `session_type`

Các biến này không thay thế normalization, nhưng giúp model hiểu chuỗi đến từ regime nào. Với multi-market downstream, đây là một cải tiến hợp lý hơn là cố để model tự suy diễn regime chỉ từ K-line dynamics.

## **6\. Kiến trúc đề xuất và ý nghĩa từng phần**

### **6.1. Luồng tổng thể**

Kiến trúc nên được hiểu theo chuỗi xử lý sau:

1. Input sequence  
2. Relative/log preprocessing  
3. Tokenizer rời rạc hóa K-line  
4. Hierarchical token sequence  
5. Decoder-only autoregressive Transformer  
6. Task head downstream hoặc mapping forecast  
7. Regression output

Theo repo chính thức, `KronosPredictor` xử lý preprocessing, normalization, prediction và inverse normalization trong cùng pipeline suy luận. Vì vậy, khi mô tả backbone, cần nhấn mạnh rằng Kronos không chỉ là “Transformer cho time series”, mà là một pipeline tokenization \+ autoregressive modeling.

### **6.2. Input layer**

Input layer nhận tensor hoặc bảng lịch sử K-line theo từng symbol:

X∈RB×L×F

Trong đó:

* `B`: batch size  
* `L`: sequence length  
* `F`: số biến K-line đã chuẩn hóa

Với bài toán hiện tại, `seq_len = 64` là điểm khởi đầu hợp lý và cũng khớp với export downstream hiện có trong repo. Ngoài ra, model Kronos-small và Kronos-base có max context 512, nên `seq_len = 64` nằm rất an toàn trong giới hạn xử lý của backbone.

### **6.3. Tokenizer K-line**

Đây là phần quan trọng nhất của Kronos. Tokenizer nhận dữ liệu K-line liên tục và ánh xạ thành token sequence, bảo toàn cả price dynamics lẫn trade activity patterns. Trong downstream, đây cũng là nơi nhạy nhất với distribution shift. Nếu đầu vào bị chuẩn hóa quá mạnh hoặc bị biến dạng quá xa khỏi K-line semantics, tokenizer sẽ khó giữ được lợi thế pretrain. Vì vậy, mục tiêu đúng không phải là làm dữ liệu đẹp nhất về mặt thống kê, mà là giữ đầu vào đủ ổn định nhưng vẫn đủ giống miền K-line tài chính thật.

### **6.4. Hierarchical tokens**

Paper Kronos mô tả tokenization theo hướng phân cấp coarse-to-fine. Ý nghĩa của bước này là cho phép mô hình biểu diễn market record ở nhiều mức chi tiết khác nhau: phần coarse nắm cấu trúc chính, phần fine nắm residual chi tiết hơn. Đây là lý do Kronos phù hợp với dữ liệu giá tài chính vốn có nhiều chuyển động nhỏ xen lẫn các regime shift và jump hiếm gặp.

### **6.5. Autoregressive Transformer blocks**

Sau bước tokenization, predictor dùng decoder-only autoregressive Transformer để học chuỗi token theo thời gian. Đây là phần trực tiếp học temporal dependency của market dynamics. Khác với iTransformer hay PatchTST, Kronos không bắt đầu từ một embedding liên tục của feature tensor downstream rồi đi thẳng vào forecasting head; nó học trong không gian token đã được tokenizer tài chính chuyên biệt xây dựng.

### **6.6. Projection head**

Ở downstream forecasting cho repo hiện tại, output cuối cùng vẫn cần được ánh xạ về dự báo liên tục phục vụ `target_ret_5d`. Nói cách khác, dù backbone bên trong là token autoregression, pipeline cuối vẫn phải sinh ra một scalar hoặc một mapping dự báo đủ để chuẩn hóa về prediction contract của dự án. Đây là phần task-level mapping, không làm thay đổi bản chất backbone của Kronos.

## **7\. Thiết kế output và objective**

### **7.1. Output khuyến nghị**

Output nên là một scalar regression:

yi,t​ ∈ R 

trong đó:

yi,t​ ≈ yi,t​=target\_ret\_5d(i,t)  

Đây là thiết kế gọn, đúng contract hiện tại, và dễ so sánh với các model khác.

### **7.2. Objective**

Khuyến nghị dùng Huber loss cho regression:

![][image1]

Lý do là return tài chính thường có outlier và tail dày, nên Huber thường ổn hơn MSE thuần. Có thể benchmark thêm với MAE hoặc MSE, nhưng Huber nên là objective mặc định ban đầu. Trong case Kronos, phần objective downstream này là lớp task-level ở trên cùng; nó không mâu thuẫn với việc backbone bên dưới vận hành theo cơ chế token autoregression.

### **7.3. Prediction schema**

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

### **8.1. Không chỉ nhìn loss**

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

Cách đánh giá này sát hơn với use case stock selection so với chỉ tối ưu sai số số học. Riêng với Kronos, vì backbone là autoregressive generative model, nên ngoài point forecast, có thể cân nhắc multi-sample forecast để làm ổn định estimate trước khi tính metric downstream.

### **8.2. Validation strategy**

Nên giữ split theo thời gian như hiện tại, không shuffle theo ngày giữa train, valid, test. Early stopping nên theo một metric tổng hợp, ưu tiên valid Huber loss kết hợp valid IC. Điều này giúp tránh chọn model chỉ khớp biên độ mà không có giá trị xếp hạng. Với Kronos, nếu mở finetune tokenizer, cần theo dõi kỹ thêm độ ổn định của tokenization downstream, vì lỗi ở tầng tokenizer có thể lan sang predictor.

## **9\. Thiết kế inference**

### **9.1. Inference offline trên test**

Cho mỗi symbol:

* lấy `seq_len` phiên gần nhất có đủ OHLCV đã chuẩn hóa  
* áp đúng logic preprocessing ở train time  
* đưa qua tokenizer và predictor  
* nhận `y_pred`  
* xuất đúng prediction contract

### **9.2. Inference production-like**

Tại ngày t:

* cập nhật raw data  
* build lại normalized OHLCV cho ngày t  
* lấy cửa sổ t−L+1 đến t  
* transform theo đúng config train-time  
* chạy model để thu được dự báo cho horizon 5 phiên

Repo chính thức cho thấy `KronosPredictor.predict()` nhận `df`, `x_timestamp`, `y_timestamp`, `pred_len`, cùng các tham số sampling như `T`, `top_p`, `sample_count`. Điều này cho phép vừa point forecast, vừa có thể average nhiều trajectory để giảm variance nếu cần.

### **9.3. Gói artifact nên lưu**

Nên lưu cùng model:

* checkpoint trọng số predictor  
* tokenizer path hoặc tokenizer checkpoint  
* `model_config.json`  
* preprocessing config cho OHLCV normalized  
* column order  
* target definition  
* training metadata  
* `run_id`

Nếu có mở finetune tokenizer, cần version hóa tokenizer riêng thay vì chỉ lưu predictor.

## **10\. Caveat quan trọng**

### **10.1. Mixed-currency**

Đây là rủi ro lớn nhất nếu dùng giá tuyệt đối. Foreign không nên bị ép về VND rồi train bằng price level. Cách đúng là dùng OHLCV theo hướng relative/return, cộng với volume đã log hoặc robust-scale.

### **10.2. Mixed-calendar**

Calendar hiện là union của nhiều thị trường. Không nên ép toàn bộ mã vào một lưới ngày giả đồng bộ rồi fill mạnh tay. Hợp lý hơn là build sequence riêng cho từng symbol. Với Kronos, đây cũng là cách tự nhiên hơn vì predictor hoạt động trên chuỗi lịch sử cụ thể của từng series.

### **10.3. Leakage do backfill**

Nếu feature engineering có `bfill` đầu chuỗi thì model có thể hưởng lợi giả tạo từ thông tin tương lai. Với Kronos, dù đầu vào giữ gần OHLCV hơn, vẫn cần tránh các thao tác làm méo temporal causality. Khuyến nghị chỉ cho phép sample sau khi cửa sổ rolling đã hình thành tự nhiên.

### **10.4. Thiếu market context**

Nếu shared pipeline chưa carry `market`, `currency`, `country`, `source`, thì model khó phân biệt regime giữa Việt Nam và foreign. Với Kronos, một hướng mở rộng hợp lý là thêm market context token thay vì chỉ dựa vào K-line dynamics để suy diễn regime.

## **11\. Cấu hình khuyến nghị để bắt đầu**

Dữ liệu

* dùng sequence theo từng symbol  
* `seq_len = 64`  
* `pred_len = 5`  
* giữ OHLCV normalized, không dùng raw absolute  
* `amount/value` chỉ dùng nếu dữ liệu sạch

Feature đầu vào

* `open_rel`  
* `high_rel`  
* `low_rel`  
* `close_ret` hoặc `log_return`  
* `volume_log`  
* `amount/value` nếu available và đáng tin cậy

Model

* `model_family = kronos`  
* ưu tiên giữ tokenizer gần pretrain  
* predictor là phần finetune chính  
* không mở rộng kiến trúc quá sớm trước khi chốt preprocessing

Finetune

* tokenizer: freeze hoặc finetune nhẹ nếu downstream chưa lệch xa  
* predictor: finetune chính cho horizon 5 phiên  
* chỉ tăng mức finetune tokenizer nếu quan sát thấy domain mismatch rõ

Loss

* Huber

Theo dõi

* valid loss  
* valid IC  
* test IC  
* top-k realized return  
* stability giữa các forecast samples nếu dùng multi-sample inference

Cấu hình này đủ gọn để benchmark nhưng vẫn bám đúng bản chất Kronos và dữ liệu hiện tại.

## **12\. Kết luận**

Kronos là lựa chọn phù hợp cho bài toán kline daily đa mã nếu dữ liệu được tổ chức đúng: mỗi sample là chuỗi lịch sử của một symbol, input giữ semantics K-line, và preprocessing ưu tiên relative/return thay vì raw absolute price. Nếu giữ nguyên giá tuyệt đối mixed-currency hoặc kéo Kronos sang một generic feature panel quá xa miền K-line, mô hình vẫn có thể train được nhưng rất dễ đánh mất lợi thế lớn nhất của backbone, tức tokenizer/codebook và distribution alignment với pretraining. Nút thắt chính không chỉ nằm ở forecasting head, mà nằm ở việc giữ đúng “ngôn ngữ K-line” trong downstream, kiểm soát mixed-market scale, và finetune tokenizer/predictor theo mức độ lệch miền thực tế.

## **13\. Tài liệu tham chiếu**

Kronos: *A Foundation Model for the Language of Financial Markets*  
arXiv: `https://arxiv.org/abs/2508.02739`

Official implementation  
GitHub: `https://github.com/Leegoo-dev/Kronos`

Finetune scripts và config tokenizer/predictor  
GitHub finetune directory và `train_tokenizer.py`

[image1]: <data:image/png;base64,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>