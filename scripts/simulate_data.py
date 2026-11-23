from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "simulated_energy_data.csv"


def generate_simulated_data() -> None:
    print("Simulating building energy data...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=24 * 100, freq="h")

    data = pd.DataFrame({"timestamp": dates})
    data["hour_of_day"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month

    # Simulate weather patterns
    data["temperature"] = (
        15
        + 10 * np.sin(data["hour_of_day"] * 2 * np.pi / 24)
        + 5 * np.sin(data["month"] * 2 * np.pi / 12)
        + np.random.randn(len(data)) * 2
    )
    data["humidity"] = (
        60
        - 15 * np.cos(data["hour_of_day"] * 2 * np.pi / 24)
        + np.random.randn(len(data)) * 5
    ).clip(0, 100)

    # Simulate occupancy (higher during weekdays 9 AM - 5 PM)
    data["occupancy"] = 0
    work_hours = (
        (data["day_of_week"] < 5)
        & (data["hour_of_day"] >= 9)
        & (data["hour_of_day"] < 17)
    )
    data.loc[work_hours, "occupancy"] = np.random.randint(50, 150, size=work_hours.sum())
    data.loc[~work_hours, "occupancy"] = np.random.randint(0, 10, size=(~work_hours).sum())

    # Simulate energy consumption (kWh)
    data["energy_kwh"] = (
        50
        + np.maximum(0, 20 - data["temperature"]) * 1.5
        + np.maximum(0, data["temperature"] - 22) * 2.0
        + data["occupancy"] * 0.1
        + np.sin(data["hour_of_day"] * 2 * np.pi / 24) * 10
        + np.random.randn(len(data)) * 5
    ).clip(lower=5)

    final_data = data[
        [
            "timestamp",
            "temperature",
            "humidity",
            "occupancy",
            "hour_of_day",
            "day_of_week",
            "energy_kwh",
        ]
    ].set_index("timestamp")

    final_data.to_csv(OUTPUT_PATH)
    print(f"Simulated data saved to {OUTPUT_PATH} (shape: {final_data.shape})")


if __name__ == "__main__":
    generate_simulated_data()