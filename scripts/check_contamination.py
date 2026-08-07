"""
Diagnostic: checks whether the very first row per city is leftover
contaminated data from the original broken AQICN geo: method (before
the pivot to OpenWeather), which may never have been deleted.
"""
import pandas as pd

df = pd.read_csv("../data/features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# The earliest row for each city
first_rows = df.sort_values("timestamp").groupby("city").first().reset_index()
print("Earliest row per city:")
print(first_rows[["city", "timestamp", "aqi", "pm25", "pm10"]].to_string())

print(f"\nEarliest timestamp overall: {df['timestamp'].min()}")
print(f"Rows within first 2 minutes of collection: "
      f"{(df['timestamp'] < df['timestamp'].min() + pd.Timedelta(minutes=2)).sum()}")
