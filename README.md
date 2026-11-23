# 🏢 SmartBEMO — Smart Building Energy Management & Optimizer

SmartBEMO is an end-to-end machine learning application that forecasts 24-hour building energy consumption (`kWh`) using live weather forecasts, logs predictions to SQLite, evaluates model accuracy (MAE/MAPE), and generates automated load-shifting recommendations.

## Features
- **ML Forecasting Engine:** LightGBM regressor tuned with `TimeSeriesSplit` and cyclical/lagged time-series features.
- **Live Weather Integration:** Fetches 5-day / 3-hour weather forecasts from OpenWeatherMap and interpolates hourly temperature and humidity.
- **Automated Advisory Rules:** Detects peak-tariff surges, overnight base-load anomalies, and daytime occupancy mismatches.
- **FastAPI Backend & Streamlit Dashboard:** Interactive dual-axis forecast charts, SQLite run history viewer, and actual-vs-predicted error evaluation.

## Project Structure
```text
SmartBEMO/
├── backend/
│   └── main.py              # FastAPI REST API & SQLite logging
├── frontend/
│   └── app.py               # Streamlit interactive dashboard
├── scripts/
│   ├── simulate_data.py     # Generates synthetic historical building dataset
│   ├── train_model.py       # Feature engineering & LightGBM hyperparameter tuning
│   └── simulate_actuals.py  # Simulates actual meter readings for evaluation
├── data/                    # Datasets & SQLite database (auto-generated)
├── models/                  # Trained .joblib artifacts (auto-generated)
├── .env.example             # Environment variable template
└── requirements.txt         # Python dependencies