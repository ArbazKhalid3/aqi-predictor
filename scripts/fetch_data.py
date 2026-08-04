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
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Real AQICN station UIDs for Pakistan, found via search_stations.py.
# Only these 4 cities get an AQICN cross-check; everyone else relies
# solely on the OpenWeather-derived AQI.
AQICN_STATION_UIDS = {
    "Karachi": 11790,
    "Lahore": 11765,
    "Islamabad": 11739,
    "Peshawar": 11791,
}

# EPA breakpoint tables: (C_low, C_high, AQI_low, AQI_high)
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
    """Piecewise-linear EPA formula: maps a pollutant concentration to a 0-500 sub-index."""
    if concentration is None:
        return None
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return round(
                ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            )
    return 500  # above scale -> cap at hazardous max


def calculate_aqi(pm25, pm10):
    """
    Overall AQI = the WORST (max) of the individual pollutant sub-indices --
    this is the real EPA methodology, not an average. Using only PM2.5/PM10
    for now since they dominate AQI in Pakistan and are the most reliably
    available; CO/NO2/SO2/O3 breakpoints can be added the same way later.
    """
    sub_indices = [
        i for i in [_sub_aqi(pm25, PM25_BREAKPOINTS), _sub_aqi(pm10, PM10_BREAKPOINTS)]
        if i is not None
    ]
    return max(sub_indices) if sub_indices else None


def fetch_openweather_pollution(lat, lon):
    """Raw pollutant concentrations (ug/m3) from OpenWeather -- our primary AQI source."""
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["list"][0]  # components + OpenWeather's own 1-5 index


def fetch_openweather_weather(lat, lon):
    """Current weather conditions from OpenWeather."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_aqicn_validation(city_name):
    """
    Real ground-station AQI, only for the 4 cities with a genuine AQICN
    station. Returns None for every other city (expected, not an error).
    """
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
    computed_aqi = calculate_aqi(components.get("pm2_5"), components.get("pm10"))
    print("PM2.5:", components.get("pm2_5"), "  PM10:", components.get("pm10"))
    print("Computed EPA AQI:", computed_aqi)
    print("OpenWeather's own 1-5 index:", pollution["main"]["aqi"])

    real_aqi = fetch_aqicn_validation(test_city["city"])
    print("AQICN ground-station AQI (validation):", real_aqi)

    weather = fetch_openweather_weather(test_city["lat"], test_city["lon"])
    print("Temp (°C):", weather["main"]["temp"])
