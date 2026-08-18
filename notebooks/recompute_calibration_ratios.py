"""
Recomputes calibration as a RATIO (real_aqi / raw_aqi) instead of an
additive offset, using only pre-Aug-11 data (before the flawed offset-based
calibration went live, so this is genuinely raw uncalibrated data).

A ratio can't push a positive value negative/to zero the way an additive
offset can, so this avoids the Lahore/Peshawar clamping problem entirely.
"""
import pandas as pd

df = pd.read_csv("../data/features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

# Only pre-calibration data -- these "aqi" values are genuinely raw
pre_calibration = df[df["timestamp"] < "2026-08-11"]
calib_df = pre_calibration.dropna(subset=["aqi_reference"])

print("Per-city ratio (real / raw), computed from pre-calibration data:")
ratios = {}
for city in ["Karachi", "Lahore", "Islamabad", "Peshawar"]:
    city_data = calib_df[calib_df["city"] == city]
    mean_raw = city_data["aqi"].mean()
    mean_real = city_data["aqi_reference"].mean()
    ratio = mean_real / mean_raw
    ratios[city] = ratio
    print(f"  {city}: mean_raw={mean_raw:.1f}  mean_real={mean_real:.1f}  ratio={ratio:.3f}")

print(f"\nCALIBRATION_RATIOS = {ratios}")
