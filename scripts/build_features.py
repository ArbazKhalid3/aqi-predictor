"""
Feature pipeline: pulls OpenWeather data for every city (primary source,
all 30 cities), computes EPA AQI, logs AQICN validation data for the 4
reference cities, and writes to BOTH the local CSV (fast local debugging)
and the Hopsworks feature store (the real feature store per project brief).
"""
import os
import time
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
import hopsworks

from cities import CITIES
from fetch_data import (
    fetch_openweather_pollution,
    fetch_openweather_weather,
    fetch_aqicn_validation,
    calculate_aqi,
    AQICN_STATION_UIDS,
)

load_dotenv()

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
    computed_aqi = calculate_aqi(components.get("pm2_5"), components.get("pm10"), city=city)

    reference_aqi = None
    if city in AQICN_STATION_UIDS:
        try:
            reference_aqi = fetch_aqicn_validation(city)
        except Exception:
            pass

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
        "aqi": computed_aqi,
        "aqi_reference": reference_aqi,
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


def push_to_hopsworks(new_df):
    """Writes this run's rows to the Hopsworks feature store, creating the
    feature group on first run and appending (upserting) on every run after."""
    try:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        )
        fs = project.get_feature_store()

        # A stable synthetic primary key (city + timestamp) so Hopsworks
        # can tell rows apart -- required for a feature group.
        upload_df = new_df.copy()
        upload_df["event_id"] = upload_df["city"] + "_" + upload_df["timestamp"]

        # Hopsworks requires the event_time column to be an actual
        # TIMESTAMP type, not plain text -- convert just for this upload
        # (the local CSV keeps the original ISO string, unaffected).
        upload_df["timestamp"] = pd.to_datetime(upload_df["timestamp"])

        fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["event_id"],
            event_time="timestamp",
            description="Hourly AQI + weather features per Pakistani city",
            time_travel_format="HUDI",  # avoids needing the optional delta-spark dependency
        )
        fg.insert(upload_df, wait=False)
        print(f"Pushed {len(upload_df)} rows to Hopsworks feature store.")
    except Exception as e:
        # Don't let a Hopsworks hiccup break the whole pipeline -- the
        # local CSV write below still succeeds either way.
        print(f"WARNING: Hopsworks push failed, continuing with local CSV only: {e}")


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

    # Push this run's fresh rows to Hopsworks
    push_to_hopsworks(new_df)

    # Local CSV: still maintained as a fast-access backup/debugging copy
    if os.path.exists(FEATURES_CSV):
        old_df = pd.read_csv(FEATURES_CSV)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["city", "timestamp"])
    combined["aqi_change_rate"] = combined.groupby("city")["aqi"].diff()
    combined.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(new_df)} new rows locally. Total rows in {FEATURES_CSV}: {len(combined)}")


if __name__ == "__main__":
    run_pipeline()
