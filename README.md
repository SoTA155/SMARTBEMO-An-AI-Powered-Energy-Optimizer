# 🏢 SmartBEMO — Smart Building Energy Management & Optimizer

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-9cf)
![SQLite](https://img.shields.io/badge/SQLite-Logging-003B57?logo=sqlite&logoColor=white)

**SmartBEMO** is an end-to-end machine learning application that forecasts 24-hour building electricity consumption (`kWh`) using live weather forecasts, logs prediction runs to SQLite, evaluates forecast accuracy (`MAE` / `MAPE`), and generates automated energy-saving recommendations.

---

## ✨ Key Features

* **Time-Series ML Forecasting:** Hyperparameter-tuned **LightGBM** regressor (`TimeSeriesSplit` + `RandomizedSearchCV`) trained on 14 engineered features, including cyclical time encodings (`sin`/`cos`), rolling temperature averages, and autoregressive consumption lags (`1h` to `168h`).
* **Live Weather Integration:** Connects to the **OpenWeatherMap 5-Day / 3-Hour Forecast API** and performs linear hourly interpolation for temperature and humidity at any building latitude/longitude.
* **Automated Optimization Rules:** Evaluates hourly forecasts against peak-tariff windows, overnight base-load thresholds, relative demand spikes, and business-hour occupancy anomalies to provide actionable recommendations.
* **Interactive Dashboard & Evaluation:** Built with **Streamlit** and **Altair**, featuring dual-axis interactive forecast charts, a historical SQLite run viewer, and a **Predicted vs. Simulated Actuals** accuracy evaluator (`MAE` and `MAPE`).

---

## 📁 Project Structure

```text
SmartBEMO/
├── backend/
│   └── main.py                     # FastAPI REST API, feature pipeline & SQLite logging
├── frontend/
│   └── app.py                      # Streamlit UI, Altair charts & accuracy evaluator
├── scripts/
│   ├── simulate_data.py            # Generates synthetic historical building dataset
│   ├── train_model.py              # Feature engineering & LightGBM hyperparameter tuning
│   └── simulate_actuals.py         # Generates simulated actual meter readings for evaluation
├── data/
│   └── simulated_energy_data.csv   # Training dataset (auto-generated)
├── models/
│   ├── energy_predictor_model.joblib # Trained LightGBM model (auto-generated)
│   └── model_features.joblib         # Ordered feature list (auto-generated)
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation