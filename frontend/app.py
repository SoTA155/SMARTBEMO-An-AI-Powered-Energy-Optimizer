from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

import altair as alt
from dotenv import load_dotenv
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import streamlit as st

load_dotenv()

# --- Configuration ---
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PREDICT_API_URL = f"{BACKEND_BASE_URL}/predict/"
RECOMMEND_API_URL = f"{BACKEND_BASE_URL}/recommendations/"

DAYS_OF_WEEK = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SQLITE_DB_PATH = DATA_DIR / "smartbemo_log.db"
ACTUAL_LOG_PATH = DATA_DIR / "actual_consumption_log.csv"

# --- Page Setup ---
st.set_page_config(page_title="SmartBEMO Energy Advisor", page_icon="🏢", layout="wide")
st.title("🏢 SmartBEMO: Building Energy Predictor & Optimizer")
st.markdown(
    "Forecast 24-hour building energy consumption using live weather data, receive actionable optimization insights, and evaluate historical accuracy."
)

# --- Sidebar Inputs ---
st.sidebar.header("Building & Simulation Parameters")
latitude = st.sidebar.number_input("Building Latitude", value=51.5074, format="%.4f")
longitude = st.sidebar.number_input("Building Longitude", value=-0.1278, format="%.4f")
occupancy = st.sidebar.number_input(
    "Expected Occupancy (People)", min_value=0, max_value=500, value=50, step=10
)
hour_of_day = st.sidebar.slider("Forecast Start Hour (0-23)", min_value=0, max_value=23, value=14)
day_name = st.sidebar.selectbox(
    "Forecast Start Day", options=list(DAYS_OF_WEEK.values()), index=1
)
day_of_week = [k for k, v in DAYS_OF_WEEK.items() if v == day_name][0]

results_container = st.container()

# --- Forecast & Recommendation Execution ---
if st.sidebar.button("Generate Forecast & Recommendations", type="primary"):
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "occupancy": occupancy,
        "start_hour_of_day": int(hour_of_day),
        "start_day_of_week": day_of_week,
    }

    with results_container:
        st.subheader("1. 24-Hour Energy & Weather Forecast")
        c1, c2, c3 = st.columns(3)
        c1.metric("Coordinates", f"{latitude:.3f}, {longitude:.3f}")
        c2.metric("Occupancy", f"{occupancy} people")
        c3.metric("Start Window", f"{day_name}, {int(hour_of_day):02d}:00 UTC")

        try:
            res_pred = requests.post(PREDICT_API_URL, json=payload, timeout=30)
            res_pred.raise_for_status()
            pred_json = res_pred.json()
            forecast_data = pred_json.get("forecast", [])
            city_name = pred_json.get("city_name", "N/A")

            if not forecast_data:
                st.warning("API returned an empty forecast list.")
            else:
                st.success(f"Forecast generated using live weather data for **{city_name}**.")
                forecast_df = pd.DataFrame(forecast_data)
                forecast_df["datetime"] = pd.to_datetime(
                    forecast_df["dt_timestamp"], unit="s", utc=True
                )
                forecast_df["time_label"] = forecast_df["datetime"].dt.strftime("%a %H:%M")

                df_melt = forecast_df.melt(
                    id_vars=["time_label", "datetime"],
                    value_vars=["predicted_kwh", "forecast_temp", "forecast_humidity"],
                    var_name="Metric",
                    value_name="Value",
                )

                base = alt.Chart(df_melt).encode(
                    x=alt.X("datetime:T", axis=alt.Axis(title="Forecast Time (UTC)", format="%a %H:%M")),
                    tooltip=["time_label", "Metric", alt.Tooltip("Value:Q", format=".1f")],
                )

                line_energy = (
                    base.transform_filter(alt.datum.Metric == "predicted_kwh")
                    .mark_line(point=True, color="#1f77b4")
                    .encode(y=alt.Y("Value:Q", axis=alt.Axis(title="Energy (kWh)", titleColor="#1f77b4")))
                )
                line_temp = (
                    base.transform_filter(alt.datum.Metric == "forecast_temp")
                    .mark_line(point=True, color="#d62728", strokeDash=[4, 4])
                    .encode(y=alt.Y("Value:Q", axis=alt.Axis(title="Temp (°C)", titleColor="#d62728")))
                )
                line_humidity = (
                    base.transform_filter(alt.datum.Metric == "forecast_humidity")
                    .mark_line(point=True, color="#2ca02c", strokeDash=[2, 2])
                    .encode(y=alt.Y("Value:Q", axis=alt.Axis(title="Humidity (%)", titleColor="#2ca02c")))
                )

                chart = (
                    alt.layer(line_energy, line_temp, line_humidity)
                    .resolve_scale(y="independent")
                    .properties(height=380)
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")

                # Fetch Recommendations
                st.subheader("2. Optimization Recommendations")
                res_rec = requests.post(
                    RECOMMEND_API_URL, json={"forecast": forecast_data}, timeout=15
                )
                res_rec.raise_for_status()
                for rec in res_rec.json().get("recommendations", []):
                    st.info(f"💡 {rec}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to FastAPI backend at `{BACKEND_BASE_URL}`. Make sure `uvicorn` is running."
            )
        except Exception as e:
            st.error(f"Error generating forecast: {e}")

# --- Historical Prediction Logs (SQLite) ---
st.markdown("---")
st.header("📜 Historical Prediction Logs")

selected_run_id = None
selected_log_df = None
selected_prediction_time_pd = None

if not SQLITE_DB_PATH.exists():
    st.info("No database found yet. Run a prediction above to initialize the log.")
else:
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            runs_df = pd.read_sql_query(
                "SELECT * FROM prediction_runs ORDER BY prediction_timestamp_utc DESC LIMIT 100",
                conn,
            )
            if runs_df.empty:
                st.warning("No prediction runs recorded yet.")
            else:
                run_map = {
                    f"Run #{row['id']} — {row['prediction_timestamp_utc']} UTC (Occ: {row['request_occupancy']})": row["id"]
                    for _, row in runs_df.iterrows()
                }
                selected_label = st.selectbox("Select Prediction Run:", options=list(run_map.keys()))
                selected_run_id = run_map[selected_label]

                run_row = runs_df[runs_df["id"] == selected_run_id].iloc[0]
                selected_prediction_time_pd = pd.to_datetime(run_row["prediction_timestamp_utc"])

                selected_log_df = pd.read_sql_query(
                    "SELECT * FROM forecast_details WHERE run_id = ?",
                    conn,
                    params=(int(selected_run_id),),
                )
                if not selected_log_df.empty:
                    selected_log_df["forecast_dt"] = pd.to_datetime(
                        selected_log_df["forecast_dt_timestamp"], unit="s", utc=True
                    )
                    st.dataframe(
                        selected_log_df[
                            [
                                "forecast_dt",
                                "predicted_kwh",
                                "forecast_temp",
                                "forecast_humidity",
                                "forecast_hour_of_day",
                                "forecast_day_of_week",
                            ]
                        ].rename(columns={"forecast_dt": "Forecast Time (UTC)"}),
                        width="stretch",
                    )
    except Exception as e:
        st.error(f"Database read error: {e}")

# --- Actuals vs. Predicted Comparison ---
st.markdown("---")
st.header("📊 Predicted vs. Simulated Actuals Comparison")

if selected_log_df is not None and not selected_log_df.empty:
    if not ACTUAL_LOG_PATH.exists():
        st.info(
            "Actuals log not found. Run `python scripts/simulate_actuals.py` to generate simulated actual consumption data."
        )
    else:
        try:
            actual_df = pd.read_csv(ACTUAL_LOG_PATH)
            actual_df["prediction_timestamp_utc"] = pd.to_datetime(
                actual_df["prediction_timestamp_utc"]
            )
            run_actuals = actual_df[
                actual_df["prediction_timestamp_utc"] == selected_prediction_time_pd
            ]

            if run_actuals.empty:
                st.warning(
                    "No simulated actuals found for this run. Re-run `python scripts/simulate_actuals.py` to update."
                )
            else:
                comp_df = pd.merge(
                    selected_log_df[["forecast_dt_timestamp", "predicted_kwh", "forecast_dt"]],
                    run_actuals[["forecast_dt_timestamp", "actual_kwh"]],
                    on="forecast_dt_timestamp",
                    how="inner",
                )
                if not comp_df.empty:
                    mae = mean_absolute_error(comp_df["actual_kwh"], comp_df["predicted_kwh"])
                    mape = mean_absolute_percentage_error(
                        comp_df["actual_kwh"], comp_df["predicted_kwh"]
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} kWh")
                    m2.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.1%}")

                    comp_melt = comp_df.melt(
                        id_vars=["forecast_dt"],
                        value_vars=["predicted_kwh", "actual_kwh"],
                        var_name="Series",
                        value_name="kWh",
                    )
                    comp_chart = (
                        alt.Chart(comp_melt)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("forecast_dt:T", axis=alt.Axis(title="Time (UTC)", format="%a %H:%M")),
                            y=alt.Y("kWh:Q", axis=alt.Axis(title="Energy (kWh)")),
                            color="Series:N",
                            tooltip=["forecast_dt:T", "Series:N", "kWh:Q"],
                        )
                        .properties(height=320)
                        .interactive()
                    )
                    st.altair_chart(comp_chart, width="stretch")
        except Exception as e:
            st.error(f"Error comparing actuals: {e}")