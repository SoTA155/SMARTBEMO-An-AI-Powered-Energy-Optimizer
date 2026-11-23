# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import math
import numpy as np
from scipy.stats import randint, uniform # For parameter distributions
from pathlib import Path # Use pathlib for paths

print("Starting model training (LightGBM with Hyperparameter Tuning)...")

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent # Project Root
DATA_PATH = BASE_DIR / "data" / "simulated_energy_data.csv" # Use Path
MODEL_DIR = BASE_DIR / "models" # Use Path
MODEL_PATH = MODEL_DIR / 'energy_predictor_model.joblib' # Use Path

BASE_FEATURES = ['temperature', 'humidity', 'occupancy', 'hour_of_day', 'day_of_week']
LAG_FEATURES = []
CYCLICAL_FEATURES = []
ROLLING_FEATURES = []
FINAL_FEATURES = []
TARGET = 'energy_kwh'
TEST_SIZE = 0.2
N_ITER_SEARCH = 20 # Number of tuning iterations (increase for better search)
CV_SPLITS = 5 # Number of time series folds
RANDOM_STATE = 42

# --- Load Data ---
try:
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}. Run simulate_data.py first.")
        exit()
    df = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True); df.sort_index(inplace=True)
    print(f"Data loaded: {DATA_PATH}, shape: {df.shape}")
except Exception as e: print(f"Error loading data: {e}"); exit()

# --- Feature Engineering ---
print("Performing feature engineering...")
lags = [1, 2, 3, 24, 48, 168]; LAG_FEATURES = []
for lag in lags: feature_name = f'{TARGET}_lag_{lag}'; df[feature_name] = df[TARGET].shift(lag); LAG_FEATURES.append(feature_name)
df['hour_sin'] = np.sin(2*np.pi*df['hour_of_day']/24.0); df['hour_cos'] = np.cos(2*np.pi*df['hour_of_day']/24.0)
df['day_sin'] = np.sin(2*np.pi*df['day_of_week']/7.0); df['day_cos'] = np.cos(2*np.pi*df['day_of_week']/7.0)
CYCLICAL_FEATURES = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
df['temp_roll_mean_3h'] = df['temperature'].rolling(window=3, min_periods=1).mean(); ROLLING_FEATURES = ['temp_roll_mean_3h']
FINAL_FEATURES = ['temperature', 'humidity', 'occupancy'] + LAG_FEATURES + CYCLICAL_FEATURES + ROLLING_FEATURES
print(f"Final features ({len(FINAL_FEATURES)}): {FINAL_FEATURES}")

# --- Preprocessing (Drop NAs) ---
initial_rows = df.shape[0]; df.dropna(subset=FINAL_FEATURES + [TARGET], inplace=True); rows_after_na = df.shape[0]
print(f"\nData shape after NA drop: {rows_after_na} (dropped {initial_rows - rows_after_na} rows)")
if df.empty: print("Error: No data left after NA removal."); exit()

X = df[FINAL_FEATURES]
y = df[TARGET]

# --- Train/Test Split (CHRONOLOGICAL SPLIT for FINAL evaluation) ---
split_index = int(len(df) * (1 - TEST_SIZE))
X_train_full = X[:split_index]
X_test = X[split_index:]
y_train_full = y[:split_index]
y_test = y[split_index:]
print(f"\nUsing chronological split: Train={len(X_train_full)}, Test={len(X_test)}")

# --- Hyperparameter Tuning Setup ---
print(f"\nSetting up Hyperparameter Tuning (RandomizedSearchCV)...")
lgbm = lgb.LGBMRegressor(objective='regression_l1', metric='mae', verbose=-1, n_jobs=-1, seed=RANDOM_STATE, boosting_type='gbdt')
param_dist = { 'n_estimators': randint(100, 1500), 'learning_rate': uniform(0.01, 0.2), 'num_leaves': randint(20, 60), 'feature_fraction': uniform(0.6, 0.4), 'bagging_fraction': uniform(0.6, 0.4), 'bagging_freq': randint(1, 7), 'reg_alpha': uniform(0, 1), 'reg_lambda': uniform(0, 1), 'min_child_samples': randint(5, 50) }
tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=0)
random_search = RandomizedSearchCV(estimator=lgbm, param_distributions=param_dist, n_iter=N_ITER_SEARCH, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1, verbose=2, random_state=RANDOM_STATE)

# --- Run Hyperparameter Tuning ---
print("Running Randomized Search CV...")
random_search.fit(X_train_full, y_train_full) # Fit on the full training data set
print("\nHyperparameter tuning complete.")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best CV score (Negative MAE): {random_search.best_score_:.4f} -> (Positive MAE): {-random_search.best_score_:.4f}")

# --- Train Final Model with Best Parameters ---
print("\nTraining final model using best parameters...")
best_model = random_search.best_estimator_ # Already refitted on X_train_full by default

# --- Evaluation on Held-Out Test Set ---
print("\nEvaluating final model on TEST set...")
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred); mse = mean_squared_error(y_test, y_pred); rmse = math.sqrt(mse)
print(f"\nFinal Model Evaluation:")
print(f"  MAE: {mae:.2f} kWh"); print(f"  RMSE: {rmse:.2f} kWh")

# --- Save Final Model and Features ---
MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists using pathlib
joblib.dump(best_model, MODEL_PATH)
print(f"\nFinal tuned model saved successfully to {MODEL_PATH}")
feature_list_path = MODEL_DIR / 'model_features.joblib'
joblib.dump(FINAL_FEATURES, feature_list_path)
print(f"Feature list saved successfully to {feature_list_path}")

# --- Feature Importance ---
try:
    importance_df = pd.DataFrame({'feature': best_model.feature_name_, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances (Final Model):"); print(importance_df.head(10))
except Exception as fe_err: print(f"\nCould not display feature importance: {fe_err}")