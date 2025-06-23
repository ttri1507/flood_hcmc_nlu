# main.py
import argparse
import pandas as pd
import os
from flood_prediction import FloodPredictionModel, save_results, plot_predictions

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flood Prediction AI Module')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input CSV file containing preprocessed flood data')
    parser.add_argument('--output_dir', type=str, default='flood_prediction_results', 
                        help='Directory to save output files (default: flood_prediction_results)')
    parser.add_argument('--train_end_date', type=str, default='2021-06-30',
                        help='End date for training data in YYYY-MM-DD format (default: 2021-06-30)')
    parser.add_argument('--test_end_date', type=str, default='2022-12-31',
                        help='End date for test data in YYYY-MM-DD format (default: 2022-12-31)')
    parser.add_argument('--future_end_date', type=str, default='2023-12-31',
                        help='End date for future predictions in YYYY-MM-DD format (default: 2023-12-31)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials for hyperparameter optimization (default: 50)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Tạo thư mục đầu ra
    output_folder = args.output_dir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Định nghĩa file đầu ra
    results_file = os.path.join(output_folder, 'prediction_results.txt')
    predict_file = os.path.join(output_folder, 'predict_flood_2023.csv')
    combined_file = os.path.join(output_folder, 'combined_flood_data.csv')
    result_predict_file = os.path.join(output_folder, 'result_predict.csv')

    # Đọc dữ liệu
    try:
        data = pd.read_csv(args.input_file)
        data['Ngày'] = pd.to_datetime(data['Ngày'], format='%d/%m/%Y')
        print("Danh sách cột trong dữ liệu:", data.columns.tolist())
        print("Số dòng dữ liệu gốc:", len(data))
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return

    # Xử lý từng mục tiêu
    targets = ['H Phú An Max (m)', 'H Nhà Bè Max (m)']
    results = {}
    test_dfs = []
    predict_dfs = []

    for target in targets:
        print(f"\nĐang xử lý {target}...")
        try:
            model = FloodPredictionModel(target, n_trials=args.n_trials)
            test_dates, y_test, y_pred_test, processed_df, historical_avg, metrics = model.train(
                data, pd.to_datetime(args.train_end_date), pd.to_datetime(args.test_end_date)
            )
            future_dates, predictions_2023 = model.predict_future(processed_df, historical_avg, args.future_end_date)

            # Lưu mô hình
            model.save_model(os.path.join(output_folder, f'model_{target.replace(" (m)", "").replace(" ", "_")}.pkl'))

            # Lưu kết quả
            save_results(results_file, target, metrics, predictions_2023)
            plot_predictions(output_folder, target, test_dates, y_test, y_pred_test, future_dates, predictions_2023)

            # Tạo DataFrame
            predict_df = pd.DataFrame({
                'Ngày': future_dates,
                f'{target} Dự đoán': predictions_2023,
                'Giờ xuất hiện': processed_df['H Phú An Giờ xh'].iloc[-len(future_dates):].values
            })
            test_df = pd.DataFrame({
                'Ngày': test_dates,
                f'{target} Thực tế': y_test.values,
                f'{target} Dự đoán': y_pred_test,
                'Giờ xuất hiện': processed_df['H Phú An Giờ xh'][test_dates.index].values
            })

            results[target] = {'test_df': test_df, 'predict_df': predict_df, 'model': model}
            test_dfs.append(test_df)
            predict_dfs.append(predict_df)
        except Exception as e:
            print(f"Lỗi khi xử lý {target}: {str(e)}")
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Lỗi khi xử lý {target}: {str(e)}\n")
            continue

    # Lưu file dự đoán 2023
    if predict_dfs:
        predict_df_all = predict_dfs[0][['Ngày', 'Giờ xuất hiện']].copy()
        for target, predict_df in zip(targets, predict_dfs):
            predict_df_all[f'{target} Dự đoán'] = predict_df[f'{target} Dự đoán']
        predict_df_all.to_csv(predict_file, index=False)

    # Kết hợp dữ liệu
    combined_df = data.copy()
    combined_df['Ngày'] = pd.to_datetime(combined_df['Ngày'], errors='coerce')
    combined_df = combined_df.dropna(subset=['Ngày'])

    for target in targets:
        if target in results:
            test_df = results[target]['test_df']
            predict_df = results[target]['predict_df']
            combined_df[f'{target}_Dự đoán'] = np.nan
            for _, row in test_df.iterrows():
                combined_df.loc[combined_df['Ngày'] == row['Ngày'], f'{target}_Dự đoán'] = row[f'{target} Dự đoán']

            for _, row in predict_df.iterrows():
                if row['Ngày'] not in combined_df['Ngày'].values:
                    new_row = pd.Series(index=['Ngày', 'H Phú An Giờ xh', f'{target}_Dự đoán'])
                    new_row['Ngày'] = pd.to_datetime(row['Ngày'])
                    new_row['H Phú An Giờ xh'] = row['Giờ xuất hiện']
                    new_row[f'{target}_Dự đoán'] = row[f'{target} Dự đoán']
                    combined_df = pd.concat([combined_df, new_row.to_frame().T], ignore_index=True)

    combined_df = combined_df.sort_values('Ngày').reset_index(drop=True)
    combined_df.to_csv(combined_file, index=False)

    # Tạo file result_predict.csv
    if test_dfs and predict_dfs:
        result_df = pd.DataFrame()
        for target in targets:
            if target not in results:
                continue
            test_df = results[target]['test_df']
            predict_df = results[target]['predict_df']
            test_subset = test_df[['Ngày', f'{target} Thực tế', f'{target} Dự đoán', 'Giờ xuất hiện']].copy()
            predict_subset = predict_df[['Ngày', f'{target} Dự đoán', 'Giờ xuất hiện']].copy()
            predict_subset[f'{target} Thực tế'] = np.nan
            target_df = pd.concat([test_subset, predict_subset], ignore_index=True)
            target_df = target_df.sort_values('Ngày')
            if result_df.empty:
                result_df = target_df
            else:
                result_df = result_df.merge(target_df.drop(columns=['Giờ xuất hiện']), on='Ngày', how='outer')

        result_df = result_df.sort_values('Ngày')
        result_df.to_csv(result_predict_file, index=False)

    print(f"Đã hoàn thành! Kết quả được lưu trong folder: {output_folder}")

if __name__ == "__main__":
    main()