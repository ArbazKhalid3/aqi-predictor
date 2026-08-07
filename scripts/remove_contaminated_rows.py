"""
One-time cleanup: removes the 30 leftover rows from the original broken
AQICN geo: method (before the OpenWeather pivot) that never got deleted.
Confirmed via check_contamination.py -- all 30 rows sit within the first
76 seconds of collection with impossible duplicate pm25 values (13.0,
158.0, or 189.0 repeated across unrelated cities).
"""
import pandas as pd

FEATURES_CSV = "../data/features.csv"

df = pd.read_csv(FEATURES_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])

cutoff = df["timestamp"].min() + pd.Timedelta(minutes=2)
before_count = len(df)

df_clean = df[df["timestamp"] >= cutoff].copy()

removed = before_count - len(df_clean)
print(f"Removed {removed} contaminated rows (had {before_count}, now {len(df_clean)}).")

# aqi_change_rate depends on row order/history per city, recompute after removal
df_clean = df_clean.sort_values(["city", "timestamp"])
df_clean["aqi_change_rate"] = df_clean.groupby("city")["aqi"].diff()

df_clean.to_csv(FEATURES_CSV, index=False)
print(f"Saved cleaned data back to {FEATURES_CSV}")
