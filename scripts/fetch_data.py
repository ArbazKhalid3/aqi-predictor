"""
Raw data fetchers.

Primary source (all 30 cities): OpenWeather Air Pollution API.
  Uses atmospheric modeling, not physical stations, so it works for any
  coordinates on Earth -- no "nearest real station might be in Tibet" problem.
  Returns raw pollutant concentrations (ug/m3), not a 0-500 AQI number, so
  we compute standard EPA AQI from PM2.5/PM10 ourselves (calculate_aqi below).

Validation source (4 cities only): AQICN, using real known station UIDs
  (found via search_stations.py) instead of geo-nearest lookup. Used to
  sanity-check our computed AQI against real ground-station readings for
  Karachi, Lahore, Islamabad, Peshawar -- the only cities AQICN actually
  covers in Pakistan.

CALIBRATION HISTORY -- important, documented in the final report:
  v1 (Aug 11): additive offset correction (subtract/add a fixed amount).
    Broken: for Lahore and Peshawar, the offset exceeded the typical raw
    value, pushing corrected results negative -> clamped to 0. 70.5% of
    Lahore's readings and 17.4% of Peshawar's were invalid zeros for a week.
  v2 (Aug 18, current): RATIO correction (multiply by a scaling factor,
    computed from pre-v1 raw data). A ratio can't push a positive value to
    zero/negative the way an offset can, so this avoids the clamping bug
    structurally, not just by capping a floor.
"""
import os
import math
import requests
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

AQICN_STATION_UIDS = {
    "Karachi": 11790,
    "Lahore": 11765,
    "Islamabad": 11739,
    "Peshawar": 11791,
}

# Per-city ratio: real_aqi ~ raw_aqi * ratio. Computed from pre-calibration
# data (Aug 4-11) comparing our raw computed AQI to real AQICN readings.
CALIBRATION_RATIOS = {
    "Karachi": 2.306,
    "Lahore": 0.267,
    "Islamabad": 1.064,
    "Peshawar": 0.177,
}

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def _sub_aqi(concentration, breakpoints):
    if concentration is None:
        return None
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return round(
                ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            )
    return 500


def calculate_aqi(pm25, pm10, city=None):
    """
    Computes EPA AQI from PM2.5/PM10, then applies a per-city RATIO
    correction for the 4 validated cities (see CALIBRATION_RATIOS above).
    Cities with no ground-truth validation data are returned uncorrected --
    we do not guess a correction without evidence.
    """
    pm25_valid = pm25 is not None and not (isinstance(pm25, float) and math.isnan(pm25))
    pm10_valid = pm10 is not None and not (isinstance(pm10, float) and math.isnan(pm10))

    pm25_trunc = int(pm25 * 10) / 10 if pm25_valid else None
    pm10_trunc = int(pm10) if pm10_valid else None

    sub_indices = [
        i for i in [_sub_aqi(pm25_trunc, PM25_BREAKPOINTS), _sub_aqi(pm10_trunc, PM10_BREAKPOINTS)]
        if i is not None
    ]
    if not sub_indices:
        return None

    raw_aqi = max(sub_indices)

    if city in CALIBRATION_RATIOS:
        corrected = raw_aqi * CALIBRATION_RATIOS[city]
        return min(500, round(corrected))  # cap at EPA max, ratio can't go negative

    return raw_aqi


def fetch_openweather_pollution(lat, lon):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["list"][0]


def fetch_openweather_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_aqicn_validation(city_name):
    uid = AQICN_STATION_UIDS.get(city_name)
    if uid is None:
        return None
    url = f"https://api.waqi.info/feed/@{uid}/"
    resp = requests.get(url, params={"token": AQICN_TOKEN}, timeout=10).json()
    if resp.get("status") != "ok":
        return None
    return resp["data"].get("aqi")


if __name__ == "__main__":
    from cities import CITIES

    test_city = CITIES[0]  # Karachi
    print(f"Testing fetch for {test_city['city']}...\n")

    pollution = fetch_openweather_pollution(test_city["lat"], test_city["lon"])
    components = pollution["components"]
    computed_aqi = calculate_aqi(components.get("pm2_5"), components.get("pm10"), city=test_city["city"])
    print("PM2.5:", components.get("pm2_5"), "  PM10:", components.get("pm10"))
    print("Computed EPA AQI (ratio-calibrated):", computed_aqi)

    real_aqi = fetch_aqicn_validation(test_city["city"])
    print("AQICN ground-station AQI (validation):", real_aqi)
