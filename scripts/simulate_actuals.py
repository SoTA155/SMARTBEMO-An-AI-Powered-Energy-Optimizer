# scripts/simulate_actuals.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sqlite3
import traceback

print("Simulating actual consumption data based on SQLite prediction log...")

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = BASE_DIR / "data" / "smartbemo_log.db"
ACTUAL_LOG_PATH = BASE_DIR / "data" / "actual_consumption_log.csv" # Output CSV
NOISE_FACTOR = 0.10; MIN_ACTUAL_KWH = 5

# --- Load Prediction Log from SQLite Database ---
if not SQLITE_DB_PATH.exists(): print(f"Error: DB file not found: {SQLITE_DB_PATH}"); exit()
conn = None
try:
    conn = sqlite3.connect(SQLITE_DB_PATH); print(f"Connected to DB: {SQLITE_DB_PATH}")
    query = """ SELECT r.id as run_id, r.prediction_timestamp_utc, d.forecast_dt_timestamp, d.predicted_kwh
                FROM prediction_runs r JOIN forecast_details d ON r.id = d.run_id
                ORDER BY r.id, d.forecast_dt_timestamp; """
    pred_df = pd.read_sql_query(query, conn); print(f"Loaded {len(pred_df)} prediction details from DB.")
    if pred_df.empty: print("No details found in DB."); exit()
except Exception as e: print(f"Error loading prediction log from DB: {e}"); traceback.print_exc(); exit()
finally:
    if conn: conn.close(); print("DB connection closed.")

# --- Simulate Actual Consumption ---
print("Simulating actual kWh values...")
noise = np.random.uniform(-NOISE_FACTOR, NOISE_FACTOR, size=len(pred_df))
actual_kwh = pred_df['predicted_kwh'] * (1 + noise)
actual_kwh = actual_kwh.clip(lower=MIN_ACTUAL_KWH)
actual_df = pred_df[['run_id', 'prediction_timestamp_utc', 'forecast_dt_timestamp']].copy()
actual_df['actual_kwh'] = actual_kwh.round(2)
# Convert prediction timestamp back to datetime for consistency in CSV if needed
actual_df['prediction_timestamp_utc'] = pd.to_datetime(actual_df['prediction_timestamp_utc'])

# --- Save Actuals Log (Overwrite) ---
try:
    print(f"Saving simulated actuals to: {ACTUAL_LOG_PATH}")
    ACTUAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    actual_df.to_csv(ACTUAL_LOG_PATH, index=False, encoding='utf-8')
    print(f"Simulated actuals **overwritten** at {ACTUAL_LOG_PATH}")
    print("Sample of generated actuals:"); print(actual_df.head())
except Exception as e: print(f"Error saving actuals log: {e}"); traceback.print_exc()