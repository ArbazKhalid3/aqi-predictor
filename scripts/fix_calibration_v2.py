"""
One-time cleanup: recomputes 'aqi' for every row using the NEW ratio-based
calculate_aqi() (v2), replacing the broken additive-offset calibration (v1)
that clamped ~70% of Lahore and ~17% of Peshawar readings to an invalid 0.
No API calls needed -- pm25/pm10 are already stored, just re-running the
math with the correct calibration method for all rows, city-aware this time.
"""
import pandas as pd
from fetch_data import calculate_aqi

FEATURES_CSV = "../data/features.csv"

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} rows.")

old_aqi = df["aqi"].copy()
df["aqi"] = df.apply(lambda row: calculate_aqi(row["pm25"], row["pm10"], city=row["city"]), axis=1)

changed = (old_aqi != df["aqi"]).sum()
print(f"Recomputed AQI for all rows. {changed} rows changed.")

# Confirm the Lahore/Peshawar fix specifically
for city in ["Lahore", "Peshawar"]:
    city_mask = df["city"] == city
    zero_count = (df.loc[city_mask, "aqi"] == 0).sum()
    print(f"{city}: {zero_count} rows still at AQI=0 out of {city_mask.sum()} total")

# Recompute aqi_change_rate since it depends on aqi
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
df = df.sort_values(["city", "timestamp"])
df["aqi_change_rate"] = df.groupby("city")["aqi"].diff()
df["timestamp"] = df["timestamp"].astype(str)  # keep CSV format consistent

df.to_csv(FEATURES_CSV, index=False)
print(f"\nSaved corrected data back to {FEATURES_CSV}")
