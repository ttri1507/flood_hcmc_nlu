# Flood Prediction AI Module

## Mô tả dự án

Dự án này là một phần của **Đề tài Ngập triều** với nội dung:  
**Nội dung 6**: Xây dựng mô hình AI dự báo vị trí, thời gian, độ sâu ngập triều cho vùng sản xuất nông nghiệp trên địa bàn thành phố.  
**Công việc 6.4**: Xây dựng mô hình AI dự báo ngập triều.  

Module AI này được thiết kế để dự báo mực nước triều tối đa tại các trạm quan trắc (Phú An và Nhà Bè) trên địa bàn Thành phố Hồ Chí Minh, sử dụng mô hình **XGBoost** với tối ưu hóa tham số bằng **Optuna**. Mô hình phân tích dữ liệu lịch sử mực nước triều, tạo các đặc trưng thời gian, lag, và xu hướng để dự đoán mực nước từ năm 2020-2022 và dự báo cho năm 2023.

## Tính năng chính
- Xử lý dữ liệu thời gian với các đặc trưng như chu kỳ ngày/năm, lag, rolling mean/std, và xu hướng.
- Huấn luyện mô hình XGBoost với tối ưu hóa tham số bằng Optuna.
- Dự đoán mực nước triều cho tương lai (năm 2023) với điều chỉnh chu kỳ và biến động lịch sử.
- Lưu kết quả dự đoán dưới dạng CSV và biểu đồ PNG.
- Hỗ trợ lưu/tải mô hình đã huấn luyện để tái sử dụng.
- Giao diện command-line sử dụng `argparse` để cấu hình linh hoạt.

## Cấu trúc thư mục
```
flood_hcmc_nlu/
├── flood_prediction/
│   ├── __init__.py
│   ├── model.py       # Chứa lớp FloodPredictionModel và DataProcessor
│   ├── utils.py       # Chứa hàm tiện ích (lưu kết quả, vẽ biểu đồ)
├── main.py            # Script chính để chạy module
├── requirements.txt   # Danh sách thư viện cần cài
├── README.md          # Tài liệu hướng dẫn
└── preprocessed_flood_data3.csv  # File dữ liệu 
```

## Yêu cầu hệ thống
- Python 3.7 hoặc cao hơn
- Các thư viện Python (xem `requirements.txt`):
  - pandas>=1.5.0
  - xgboost>=1.7.0
  - scikit-learn>=1.2.0
  - numpy>=1.23.0
  - matplotlib>=3.6.0
  - optuna>=3.0.0
- Hệ điều hành: Linux, macOS, hoặc Windows (với WSL)

## Cài đặt
1. Clone repository về máy:
   ```bash
   git clone <repository_url>
   cd flood_hcmc_nlu
   ```

2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

3. Chuẩn bị file dữ liệu đầu vào (`preprocessed_flood_data3.csv`):
   - File CSV phải chứa các cột:
     - `Ngày`: Ngày tháng (định dạng `DD/MM/YYYY`)
     - `H Phú An Max (m)`: Mực nước triều tối đa tại trạm Phú An
     - `H Nhà Bè Max (m)`: Mực nước triều tối đa tại trạm Nhà Bè
     - `H Phú An Giờ xh`: Giờ xuất hiện mực nước triều tối đa tại Phú An
   - Nếu file nằm trên Google Drive, tải về bằng `wget`:
     ```bash
     wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=YOUR_FILE_ID' -O preprocessed_flood_data3.csv
     ```
     Thay `YOUR_FILE_ID` bằng ID của file từ link chia sẻ công khai.

## Cách sử dụng
### Chạy module từ command line
Chạy script `main.py` với các tham số cấu hình:

```bash
python3 main.py --input_file preprocessed_flood_data3.csv --n_trials 50 --output_dir results --train_end_date 2021-06-30 --test_end_date 2022-12-31 --future_end_date 2023-12-31
```

#### Các tham số
- `--input_file` (bắt buộc): Đường dẫn đến file CSV đầu vào.
- `--output_dir` (tùy chọn): Thư mục lưu kết quả (mặc định: `flood_prediction_results`).
- `--n_trials` (tùy chọn): Số vòng lặp tối ưu hóa tham số bằng Optuna (mặc định: 50).
- `--train_end_date` (tùy chọn): Ngày kết thúc dữ liệu huấn luyện (mặc định: `2021-06-30`).
- `--test_end_date` (tùy chọn): Ngày kết thúc dữ liệu kiểm tra (mặc định: `2022-12-31`).
- `--future_end_date` (tùy chọn): Ngày kết thúc dự đoán tương lai (mặc định: `2023-12-31`).

#### File đầu ra
- `prediction_results.txt`: Kết quả đánh giá mô hình (MSE, sai số trung bình, tỷ lệ sai lệch).
- `predict_flood_2023.csv`: Dự đoán mực nước triều năm 2023.
- `combined_flood_data.csv`: Dữ liệu gốc kết hợp với dự đoán.
- `result_predict.csv`: So sánh thực tế và dự đoán từ 2020-2022, cùng dự báo 2023.
- `prediction_plot_*.png`: Biểu đồ dự đoán cho từng trạm.

### Sử dụng module trong script Python
```python
from flood_prediction.model import FloodPredictionModel
import pandas as pd

# read data
data = pd.read_csv('preprocessed_flood_data3.csv')
data['Ngày'] = pd.to_datetime(data['Ngày'], format='%d/%m/%Y')

# Khởi tạo và huấn luyện mô hình
model = FloodPredictionModel(target_col='H Phú An Max (m)', n_trials=50)
test_dates, y_test, y_pred_test, processed_df, historical_avg, metrics = model.train(
    data, pd.to_datetime('2021-06-30'), pd.to_datetime('2022-12-31')
)

# predict
future_dates, predictions_2023 = model.predict_future(processed_df, historical_avg, '2023-12-31')

# fave model
model.save_model('model_phu_an.pkl')

# lpad model
model.load_model('model_phu_an.pkl')
```

## Ghi chú
- **Dữ liệu đầu vào**: File CSV phải có định dạng đúng và chứa đủ các cột yêu cầu. Kiểm tra dữ liệu bằng:
  ```python
  import pandas as pd
  df = pd.read_csv('preprocessed_flood_data3.csv')
  print(df.columns)
  ```
- **Hiệu suất**: Giá trị `--n_trials` lớn (ví dụ: 100) sẽ cải thiện mô hình nhưng tăng thời gian chạy.
- **Tích hợp Google Drive**: Để tự động tải file dữ liệu từ Google Drive, thêm code vào `main.py` (xem tài liệu trước đó).
- **Xử lý lỗi**:
  - Nếu gặp lỗi import, kiểm tra cấu trúc thư mục và nội dung các file trong `flood_prediction/`.
  - Nếu lỗi đọc file CSV, kiểm tra định dạng ngày tháng và các cột dữ liệu.


