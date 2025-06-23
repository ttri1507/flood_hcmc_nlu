# flood_prediction/model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import optuna
import pickle
import os

class DataProcessor:
    """Lớp xử lý dữ liệu, tạo đặc trưng từ dữ liệu thô."""
    def __init__(self, target_col, feature_cols):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.safe_target_col = target_col.replace(" (m)", "")

    def prepare_data(self, df):
        """Tạo đặc trưng từ DataFrame."""
        df = df.copy()
        if self.target_col not in df.columns:
            raise ValueError(f"Cột {self.target_col} không tồn tại trong dữ liệu")

        # Thêm đặc trưng thời gian
        df['year'] = df['Ngày'].dt.year
        df['month'] = df['Ngày'].dt.month
        df['day'] = df['Ngày'].dt.day
        df['day_of_year'] = df['Ngày'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

        # Thêm đặc trưng lag và rolling
        df[f'{self.safe_target_col}_lag_1'] = df[self.target_col].shift(1)
        df[f'{self.safe_target_col}_lag_2'] = df[self.target_col].shift(2)
        df[f'{self.safe_target_col}_lag_3'] = df[self.target_col].shift(3)
        df[f'{self.safe_target_col}_rolling_mean_3'] = df[self.target_col].rolling(window=3).mean()
        df[f'{self.safe_target_col}_rolling_std_7'] = df[self.target_col].rolling(window=7).std()
        df[f'{self.safe_target_col}_diff'] = df[self.target_col] - df[f'{self.safe_target_col}_lag_1']

        # Thêm trung bình lịch sử
        historical_avg = df[df['year'].isin([2018, 2019, 2020, 2021, 2022])].groupby('day_of_year')[self.target_col].mean().to_dict()
        df['historical_daily_avg'] = df['day_of_year'].map(historical_avg)

        # Thêm xu hướng
        df[f'{self.safe_target_col}_trend_30d'] = df[f'{self.safe_target_col}_diff'].rolling(window=30).mean()

        # Xóa NaN
        df = df.dropna()
        print(f"Số dòng sau khi xử lý NaN cho {self.target_col}: {len(df)}")

        features = self.feature_cols + [
            f'{self.safe_target_col}_lag_1', f'{self.safe_target_col}_lag_2', f'{self.safe_target_col}_lag_3',
            f'{self.safe_target_col}_rolling_mean_3', f'{self.safe_target_col}_rolling_std_7',
            f'{self.safe_target_col}_diff', f'{self.safe_target_col}_trend_30d',
            'year', 'month', 'day', 'sin_day', 'cos_day', 'sin_month', 'cos_month', 'historical_daily_avg'
        ]
        X = df[features]
        y = df[self.target_col]
        return X, y, df, historical_avg

class FloodPredictionModel:
    """Lớp mô hình dự đoán mực nước lũ bằng XGBoost."""
    def __init__(self, target_col, n_trials=50):
        self.target_col = target_col
        self.n_trials = n_trials
        self.model = None
        self.best_params = None
        self.processor = DataProcessor(target_col, [target_col])
        self.safe_target_col = target_col.replace(" (m)", "")

    def objective(self, trial, X, y, train_end_date, data):
        """Hàm mục tiêu cho Optuna."""
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
        num_boost_round = trial.suggest_int('num_boost_round', 100, 1000)

        mse_scores = []
        tscv = TimeSeriesSplit(n_splits=5)
        train_data = data[data['Ngày'] <= train_end_date]
        X_train, y_train, _, _ = self.processor.prepare_data(train_data)

        if len(X_train) < 5 * tscv.get_n_splits():
            print(f"Cảnh báo: Không đủ mẫu cho {tscv.get_n_splits()} splits, chỉ có {len(X_train)} mẫu")
            return float('inf')

        for train_idx, val_idx in tscv.split(X_train):
            if len(train_idx) == 0 or len(val_idx) == 0:
                print("Tập train hoặc validation rỗng, bỏ qua split này")
                continue
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(param, dtrain, num_boost_round=num_boost_round,
                             evals=[(dval, 'eval')], early_stopping_rounds=20, verbose_eval=False)
            y_pred_val = model.predict(dval)
            mse = mean_squared_error(y_val, y_pred_val)
            mse_scores.append(mse)

        return np.mean(mse_scores) if mse_scores else float('inf')

    def train(self, data, train_end_date, test_end_date):
        """Huấn luyện mô hình."""
        X, y, processed_df, historical_avg = self.processor.prepare_data(data)
        
        # Tách dữ liệu train/test
        train_mask = processed_df['Ngày'] <= train_end_date
        test_mask = (processed_df['Ngày'] > train_end_date) & (processed_df['Ngày'] <= test_end_date)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_dates = processed_df['Ngày'][test_mask]

        # Tối ưu hóa tham số
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        objective_func = lambda trial: self.objective(trial, X, y, train_end_date, data)
        study.optimize(objective_func, n_trials=self.n_trials)

        self.best_params = study.best_params
        num_boost_round = self.best_params.pop('num_boost_round')

        # Huấn luyện mô hình cuối cùng
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        self.model = xgb.train(self.best_params, dtrain, num_boost_round=num_boost_round,
                              evals=[(dtest, 'test')], early_stopping_rounds=20, verbose_eval=False)

        # Dự đoán test
        y_pred_test = self.model.predict(dtest)
        mse = mean_squared_error(y_test, y_pred_test)
        absolute_errors = np.abs(y_test - y_pred_test)
        metrics = {
            'mse': mse,
            'average_absolute_error': np.mean(absolute_errors),
            'total_actual': np.sum(y_test),
            'total_absolute_error': np.sum(absolute_errors),
            'total_error_ratio': (np.sum(absolute_errors) / np.sum(y_test) * 100) if np.sum(y_test) != 0 else 0
        }

        return test_dates, y_test, y_pred_test, processed_df, historical_avg, metrics

    def predict_future(self, processed_df, historical_avg, future_end_date):
        """Dự đoán cho tương lai."""
        historical_values = list(processed_df[processed_df['year'] == 2022][self.target_col].values)
        recent_diff = processed_df[processed_df['year'] == 2022][f'{self.safe_target_col}_diff'].tail(30).mean()
        future_dates = pd.date_range(start='2023-01-01', end=future_end_date, freq='D')
        predictions_2023 = []

        historical_std = processed_df[self.target_col].std()
        historical_min = processed_df[self.target_col].min()
        historical_max = processed_df[self.target_col].max()

        for i, future_date in enumerate(future_dates):
            day_of_year = future_date.dayofyear
            sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
            cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
            sin_month = np.sin(2 * np.pi * future_date.month / 12)
            cos_month = np.cos(2 * np.pi * future_date.month / 12)
            historical_daily_avg = historical_avg.get(day_of_year, np.mean(list(historical_avg.values())))

            lag1 = historical_values[-1] if historical_values else historical_daily_avg
            lag2 = historical_values[-2] if len(historical_values) >= 2 else lag1
            lag3 = historical_values[-3] if len(historical_values) >= 3 else lag2

            rolling_mean_3 = np.mean(historical_values[-3:]) if len(historical_values) >= 3 else np.mean(historical_values) if historical_values else historical_daily_avg
            rolling_std_7 = np.std(historical_values[-7:]) if len(historical_values) >= 7 else historical_std
            diff = lag1 - lag2
            trend_30d = recent_diff if i < 30 else (np.mean(predictions_2023[-30:]) - np.mean(predictions_2023[-31:-1]) if i >= 31 else recent_diff)

            X_future = pd.DataFrame({
                self.target_col: [lag1],
                f'{self.safe_target_col}_lag_1': [lag1],
                f'{self.safe_target_col}_lag_2': [lag2],
                f'{self.safe_target_col}_lag_3': [lag3],
                f'{self.safe_target_col}_rolling_mean_3': [rolling_mean_3],
                f'{self.safe_target_col}_rolling_std_7': [rolling_std_7],
                f'{self.safe_target_col}_diff': [diff],
                f'{self.safe_target_col}_trend_30d': [trend_30d],
                'year': [future_date.year],
                'month': [future_date.month],
                'day': [future_date.day],
                'sin_day': [sin_day],
                'cos_day': [cos_day],
                'sin_month': [sin_month],
                'cos_month': [cos_month],
                'historical_daily_avg': [historical_daily_avg]
            })
            d_future = xgb.DMatrix(X_future)

            pred = self.model.predict(d_future)[0]
            cycle_adjustment = historical_daily_avg + (pred - historical_daily_avg) * 0.8
            noise = np.random.normal(0, historical_std * 0.05)
            pred = cycle_adjustment + noise
            pred = min(max(pred, historical_min * 0.9), historical_max * 1.1)
            predictions_2023.append(pred)
            historical_values.append(pred)

        predictions_2023 = pd.Series(predictions_2023).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        return future_dates, predictions_2023

    def save_model(self, path):
        """Lưu mô hình vào file."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        """Tải mô hình từ file."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
