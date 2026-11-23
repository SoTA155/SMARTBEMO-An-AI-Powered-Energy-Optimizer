from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import traceback
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import requests
import uvicorn

load_dotenv()

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "energy_predictor_model.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.joblib"
SIMULATED_DATA_PATH = DATA_DIR / "simulated_energy_data.csv"
SQLITE_DB_PATH = DATA_DIR / "smartbemo_log.db"

OWM_API_KEY = os.getenv("OWM_API_KEY")
OWM_URL = "https://api.openweathermap.org/data/2.5/forecast"
DEFAULT_LAT = 51.5074
DEFAULT_LON = -0.1278

# --- Load Model & Baseline Seed History ---
model = None
model_features = None

try:
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        print(f"Model and {len(model_features)} features loaded successfully.")
    else:
        print("Warning: Model or features file missing. Run scripts/train_model.py first.")
except Exception as e:
    print(f"Error loading model/features: {e}")
    traceback.print_exc()


def get_initial_lag_history() -> List[float]:
    """Load the last 168 hours of energy usage to seed autoregressive lag features."""
    if SIMULATED_DATA_PATH.exists():
        try:
            df = pd.read_csv(SIMULATED_DATA_PATH)
            if "energy_kwh" in df.columns and len(df) >= 168:
                return df["energy_kwh"].tail(168).tolist()
        except Exception as e:
            print(f"Warning: Could not read seed history from CSV: {e}")
    return [55.0] * 168


# --- Database Initialization ---
def initialize_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_timestamp_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                request_latitude REAL,
                request_longitude REAL,
                request_occupancy INTEGER,
                request_start_hour INTEGER,
                request_start_day INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                forecast_dt_timestamp INTEGER NOT NULL,
                forecast_hour_offset INTEGER,
                forecast_hour_of_day INTEGER,
                forecast_day_of_week INTEGER,
                predicted_kwh REAL,
                forecast_temp REAL,
                forecast_humidity REAL,
                FOREIGN KEY (run_id) REFERENCES prediction_runs (id)
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecast_details_run_id ON forecast_details (run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecast_details_timestamp ON forecast_details (forecast_dt_timestamp)"
        )
        conn.commit()


initialize_database(SQLITE_DB_PATH)

app = FastAPI(
    title="SmartBEMO API",
    description="Smart Building Energy Management & Optimization API",
    version="1.1.0",
)


# --- Pydantic Models ---
class PredictionInput(BaseModel):
    latitude: float = Field(DEFAULT_LAT)
    longitude: float = Field(DEFAULT_LON)
    occupancy: int = Field(..., ge=0)
    start_hour_of_day: int = Field(..., ge=0, le=23)
    start_day_of_week: int = Field(..., ge=0, le=6)


class ForecastItem(BaseModel):
    dt_timestamp: int
    predicted_kwh: float
    hour_of_day: int
    day_of_week: int
    forecast_temp: float
    forecast_humidity: float


class PredictionResponse(BaseModel):
    forecast: List[ForecastItem]
    weather_source: str = "OpenWeatherMap 5-Day / 3-Hour Forecast"
    city_name: str = "N/A"


class RecommendationInput(BaseModel):
    forecast: List[ForecastItem] = Field(..., min_length=1)


class RecommendationResponse(BaseModel):
    recommendations: List[str]


# --- Helper Functions ---
def get_next_time(current_hour: int, current_day: int) -> tuple[int, int]:
    next_hour = (current_hour + 1) % 24
    next_day = (current_day + 1) % 7 if next_hour == 0 else current_day
    return next_hour, next_day


def get_weather_forecast_3hr(lat: float, lon: float) -> Dict:
    if not OWM_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OWM_API_KEY is missing. Add it to your .env file.",
        )
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}
    try:
        response = requests.get(OWM_URL, params=params, timeout=15)
        response.raise_for_status()
        weather_data = response.json()
        if "list" not in weather_data or not isinstance(weather_data["list"], list):
            raise ValueError("Invalid response structure from OpenWeatherMap.")
        return weather_data
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="OpenWeatherMap request timed out.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {e}")


def log_predictions_to_sqlite(
    db_path: Path, prediction_request: PredictionInput, forecast_items: List[ForecastItem]
) -> None:
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_runs (
                    request_latitude, request_longitude, request_occupancy,
                    request_start_hour, request_start_day
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    prediction_request.latitude,
                    prediction_request.longitude,
                    prediction_request.occupancy,
                    prediction_request.start_hour_of_day,
                    prediction_request.start_day_of_week,
                ),
            )
            run_id = cursor.lastrowid
            cursor.execute(
                "SELECT prediction_timestamp_utc FROM prediction_runs WHERE id = ?",
                (run_id,),
            )
            run_ts_str = cursor.fetchone()[0]
            run_ts_dt = datetime.fromisoformat(run_ts_str).replace(tzinfo=timezone.utc)

            details_data = []
            for item in forecast_items:
                forecast_dt = datetime.fromtimestamp(item.dt_timestamp, tz=timezone.utc)
                offset_hours = round((forecast_dt - run_ts_dt).total_seconds() / 3600)
                details_data.append(
                    (
                        run_id,
                        item.dt_timestamp,
                        offset_hours,
                        item.hour_of_day,
                        item.day_of_week,
                        item.predicted_kwh,
                        item.forecast_temp,
                        item.forecast_humidity,
                    )
                )

            cursor.executemany(
                """
                INSERT INTO forecast_details (
                    run_id, forecast_dt_timestamp, forecast_hour_offset,
                    forecast_hour_of_day, forecast_day_of_week,
                    predicted_kwh, forecast_temp, forecast_humidity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                details_data,
            )
    except Exception as e:
        print(f"Warning: Failed to log predictions to SQLite: {e}")
        traceback.print_exc()


# --- Endpoints ---
@app.get("/")
async def read_root():
    return {"status": "online", "service": "SmartBEMO API", "version": "1.1.0"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict_energy_forecast(input_data: PredictionInput):
    if model is None or model_features is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run 'python scripts/train_model.py' first.",
        )

    weather_data = get_weather_forecast_3hr(input_data.latitude, input_data.longitude)
    forecast_list_3hr = weather_data.get("list", [])
    city_name = weather_data.get("city", {}).get("name", "N/A")

    if not forecast_list_3hr:
        raise HTTPException(status_code=502, detail="Weather API returned empty forecast.")

    weather_points = [
        {
            "datetime": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
            "temp": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
        }
        for item in forecast_list_3hr
        if "main" in item and "temp" in item["main"] and "humidity" in item["main"]
    ]
    weather_points.sort(key=lambda x: x["datetime"])

    # Align prediction start time with the first available weather window
    base_dt = weather_points[0]["datetime"].replace(
        hour=input_data.start_hour_of_day, minute=0, second=0, microsecond=0
    )
    if base_dt < weather_points[0]["datetime"]:
        base_dt += timedelta(days=1)

    forecast_results: List[ForecastItem] = []
    current_hour = input_data.start_hour_of_day
    current_day = input_data.start_day_of_week

    energy_history = get_initial_lag_history()
    temp_history: List[float] = []
    weather_idx = 0

    for hour_offset in range(24):
        target_dt = base_dt + timedelta(hours=hour_offset)
        while (
            weather_idx + 1 < len(weather_points)
            and weather_points[weather_idx + 1]["datetime"] <= target_dt
        ):
            weather_idx += 1

        if weather_idx + 1 >= len(weather_points):
            break

        p1, p2 = weather_points[weather_idx], weather_points[weather_idx + 1]
        total_sec = (p2["datetime"] - p1["datetime"]).total_seconds()
        target_sec = (target_dt - p1["datetime"]).total_seconds()

        if total_sec <= 0:
            interp_temp, interp_humidity = p1["temp"], p1["humidity"]
        else:
            ratio = max(0.0, min(1.0, target_sec / total_sec))
            interp_temp = p1["temp"] + (p2["temp"] - p1["temp"]) * ratio
            interp_humidity = p1["humidity"] + (p2["humidity"] - p1["humidity"]) * ratio

        temp_history.append(interp_temp)
        temp_roll_mean_3h = float(np.mean(temp_history[-3:]))

        # Build exact feature vector expected by the trained LightGBM model
        feature_row = {
            "temperature": interp_temp,
            "humidity": interp_humidity,
            "occupancy": input_data.occupancy,
            "hour_of_day": current_hour,
            "day_of_week": current_day,
            "hour_sin": np.sin(2 * np.pi * current_hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * current_hour / 24.0),
            "day_sin": np.sin(2 * np.pi * current_day / 7.0),
            "day_cos": np.cos(2 * np.pi * current_day / 7.0),
            "temp_roll_mean_3h": temp_roll_mean_3h,
            "energy_kwh_lag_1": energy_history[-1],
            "energy_kwh_lag_2": energy_history[-2],
            "energy_kwh_lag_3": energy_history[-3],
            "energy_kwh_lag_24": energy_history[-24],
            "energy_kwh_lag_48": energy_history[-48],
            "energy_kwh_lag_168": energy_history[-168],
        }

        input_df = pd.DataFrame([feature_row])[model_features]
        pred_kwh = round(float(model.predict(input_df)[0]), 2)
        energy_history.append(pred_kwh)

        forecast_results.append(
            ForecastItem(
                dt_timestamp=int(target_dt.timestamp()),
                predicted_kwh=pred_kwh,
                hour_of_day=current_hour,
                day_of_week=current_day,
                forecast_temp=round(interp_temp, 1),
                forecast_humidity=round(interp_humidity, 1),
            )
        )
        current_hour, current_day = get_next_time(current_hour, current_day)

    if forecast_results:
        log_predictions_to_sqlite(SQLITE_DB_PATH, input_data, forecast_results)

    return PredictionResponse(forecast=forecast_results, city_name=city_name)


@app.post("/recommendations/", response_model=RecommendationResponse)
async def get_recommendations(input_data: RecommendationInput):
    forecast_list = input_data.forecast
    avg_kwh = sum(item.predicted_kwh for item in forecast_list) / len(forecast_list)

    recommendations: List[str] = []
    seen_hours = {"night": set(), "peak": set(), "spike": set(), "low_occ": set()}

    for item in forecast_list:
        hour, day, pred_kwh = item.hour_of_day, item.day_of_week, item.predicted_kwh
        is_night = hour >= 23 or hour <= 5
        is_peak = 16 <= hour < 20
        is_work_hours = (0 <= day <= 4) and (9 <= hour < 17)

        if is_night and pred_kwh > 65 and hour not in seen_hours["night"]:
            recommendations.append(
                f"Hour {hour:02d}:00 — High overnight load ({pred_kwh:.1f} kWh). Check HVAC setback schedules and standby equipment."
            )
            seen_hours["night"].add(hour)

        elif is_peak and pred_kwh > 75 and hour not in seen_hours["peak"]:
            recommendations.append(
                f"Hour {hour:02d}:00 — Peak tariff window surge ({pred_kwh:.1f} kWh). Consider pre-cooling or shifting non-essential loads."
            )
            seen_hours["peak"].add(hour)

        elif (
            not is_night
            and not is_peak
            and pred_kwh > avg_kwh * 1.5
            and hour not in seen_hours["spike"]
        ):
            recommendations.append(
                f"Hour {hour:02d}:00 — Usage ({pred_kwh:.1f} kWh) is >50% above 24h average ({avg_kwh:.1f} kWh). Inspect equipment staging."
            )
            seen_hours["spike"].add(hour)

        elif is_work_hours and pred_kwh < avg_kwh * 0.6 and hour not in seen_hours["low_occ"]:
            recommendations.append(
                f"Hour {hour:02d}:00 — Unusually low usage ({pred_kwh:.1f} kWh) during business hours. Verify sensor/meter telemetry."
            )
            seen_hours["low_occ"].add(hour)

    if not recommendations:
        recommendations.append(
            "Energy consumption profile stays within normal operating thresholds over the next 24 hours."
        )

    return RecommendationResponse(recommendations=recommendations)


if __name__ == "__main__":
    if not OWM_API_KEY:
        print("WARNING: OWM_API_KEY not found in .env file.")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)