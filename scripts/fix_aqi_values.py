"""
One-time cleanup: recomputes the 'aqi' column for every row in the local
CSV using the CORRECTED calculate_aqi() (with proper EPA truncation).
No API calls needed -- pm25/pm10 are already stored, we're just re-running
the math. Also recomputes aqi_change_rate afterward since it depends on aqi.
"""
import pandas as pd
from fetch_data import calculate_aqi

FEATURES_CSV = "../data/features.csv"

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} rows.")

old_aqi = df["aqi"].copy()

# Recompute aqi for every row from its already-stored pm25/pm10
df["aqi"] = df.apply(lambda row: calculate_aqi(row["pm25"], row["pm10"]), axis=1)

changed = (old_aqi != df["aqi"]).sum()
print(f"Recomputed AQI for all rows. {changed} rows changed.")

if changed > 0:
    print("\nRows that changed (before -> after):")
    diff_mask = old_aqi != df["aqi"]
    comparison = pd.DataFrame({
        "city": df.loc[diff_mask, "city"],
        "timestamp": df.loc[diff_mask, "timestamp"],
        "old_aqi": old_aqi[diff_mask],
        "new_aqi": df.loc[diff_mask, "aqi"],
    })
    print(comparison.to_string())

# aqi_change_rate depends on aqi, so it must be recomputed too, in the
# same way build_features.py originally computes it (per-city diff)
df = df.sort_values(["city", "timestamp"])
df["aqi_change_rate"] = df.groupby("city")["aqi"].diff()

df.to_csv(FEATURES_CSV, index=False)
print(f"\nSaved corrected data back to {FEATURES_CSV}")
