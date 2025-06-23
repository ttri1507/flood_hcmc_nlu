# flood_prediction/utils.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def save_results(results_file, target_col, metrics, predictions_2023):
    """Lưu kết quả vào file txt."""
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Kết quả cho {target_col}:\n")
        f.write(f"Mean Squared Error (2020-2022): {metrics['mse']:.2f}\n")
        f.write(f"Trung bình sai số dự đoán (2020-2022): {metrics['average_absolute_error']:.2f} m\n")
        f.write(f"Tổng mực nước thực tế (2020-2022): {metrics['total_actual']:.2f} m\n")
        f.write(f"Tổng sai số tuyệt đối (2020-2022): {metrics['total_absolute_error']:.2f} m\n")
        f.write(f"Tỷ lệ sai lệch tổng: {metrics['total_error_ratio']:.2f}%\n")
        f.write(f"Tổng mực nước dự đoán cho 2023: {np.sum(predictions_2023):.2f} m\n")
        f.write(f"{'-'*50}\n")

def plot_predictions(output_folder, target_col, test_dates, y_test, y_pred_test, future_dates, predictions_2023):
    """Vẽ biểu đồ dự đoán."""
    safe_target_col = target_col.replace(" (m)", "")
    test_data = pd.DataFrame({'Ngày': test_dates, 'Thực tế': y_test, 'Dự đoán': y_pred_test})
    predict_2023_df = pd.DataFrame({'Ngày': future_dates, 'Dự đoán': predictions_2023})

    plt.figure(figsize=(14, 7))
    plt.plot(test_data['Ngày'], test_data['Thực tế'], marker='o', linestyle='-', color='green', label='Thực tế 2020-2022')
    plt.plot(test_data['Ngày'], test_data['Dự đoán'], marker='x', linestyle='--', color='red', label='Dự đoán 2020-2022')
    plt.plot(predict_2023_df['Ngày'], predict_2023_df['Dự đoán'], marker='o', linestyle='--', color='blue', label='Dự đoán 2023')
    plt.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--', label='Bắt đầu 2021')
    plt.axvline(x=pd.to_datetime('2022-01-01'), color='gray', linestyle='--', label='Bắt đầu 2022')
    plt.axvline(x=pd.to_datetime('2023-01-01'), color='gray', linestyle='--', label='Bắt đầu 2023')
    plt.title(f'Dự đoán và thực tế mực nước {target_col} (2020-2023)')
    plt.xlabel('Ngày')
    plt.ylabel('Mực nước (m)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f'prediction_plot_{safe_target_col}.png')
    plt.savefig(plot_path)
    plt.close()
