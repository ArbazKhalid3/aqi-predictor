"""
Feature pipeline: pulls OpenWeather data for every city (primary source,
all 30 cities), computes EPA AQI, and also logs AQICN's real station AQI
for the 4 cities that have genuine coverage (validation/calibration data).
Appends one row per city to a local CSV -- our "feature store" for now.
"""
import os
import time
from datetime import datetime, timezone
import pandas as pd

from cities import CITIES
from fetch_data import (
    fetch_openweather_pollution,
    fetch_openweather_weather,
    fetch_aqicn_validation,
    calculate_aqi,
    AQICN_STATION_UIDS,
)

FEATURES_CSV = "../data/features.csv"


def build_row(city_info):
    city, country = city_info["city"], city_info["country"]
    lat, lon = city_info["lat"], city_info["lon"]

    try:
        pollution = fetch_openweather_pollution(lat, lon)
        weather = fetch_openweather_weather(lat, lon)
    except Exception as e:
        print(f"  SKIPPED {city}: {e}")
        return None

    components = pollution["components"]
    computed_aqi = calculate_aqi(components.get("pm2_5"), components.get("pm10"))

    # Only attempt AQICN validation for cities we know have a real station --
    # avoids wasting API calls on cities we already know return NO MATCH.
    reference_aqi = None
    if city in AQICN_STATION_UIDS:
        try:
            reference_aqi = fetch_aqicn_validation(city)
        except Exception:
            pass  # validation failure shouldn't block the main pipeline

    now = datetime.now(timezone.utc)

    row = {
        "country": country,
        "city": city,
        "lat": lat,
        "lon": lon,
        "timestamp": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "aqi": computed_aqi,               # our prediction TARGET (all 30 cities)
        "aqi_reference": reference_aqi,    # real station AQI, only Karachi/Lahore/Islamabad/Peshawar -- used later for calibration, not a feature
        "pm25": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "co": components.get("co"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "o3": components.get("o3"),
        "temp_c": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
    }
    return row


def run_pipeline():
    print(f"Building features for {len(CITIES)} cities...")
    rows = []
    for city_info in CITIES:
        row = build_row(city_info)
        if row:
            rows.append(row)
            ref = f" (station: {row['aqi_reference']})" if row["aqi_reference"] is not None else ""
            print(f"  OK {city_info['city']}: AQI={row['aqi']}{ref}")
        time.sleep(1)

    if not rows:
        print("No data collected -- check your API keys.")
        return

    new_df = pd.DataFrame(rows)

    if os.path.exists(FEATURES_CSV):
        old_df = pd.read_csv(FEATURES_CSV)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["city", "timestamp"])
    combined["aqi_change_rate"] = combined.groupby("city")["aqi"].diff()

    combined.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved {len(new_df)} new rows. Total rows in {FEATURES_CSV}: {len(combined)}")


if __name__ == "__main__":
    run_pipeline()
