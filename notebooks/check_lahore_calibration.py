import pandas as pd

df = pd.read_csv("../data/features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

lahore = df[df["city"] == "Lahore"].sort_values("timestamp")
after_calibration = lahore[lahore["timestamp"] >= "2026-08-11"]

print(f"Lahore rows after calibration deployed: {len(after_calibration)}")
print(f"Rows with AQI == 0 (clamped): {(after_calibration['aqi'] == 0).sum()}")
print(f"Percentage clamped to 0: {(after_calibration['aqi'] == 0).mean() * 100:.1f}%")
print(f"\nAQI value distribution after calibration:")
print(after_calibration["aqi"].describe())
