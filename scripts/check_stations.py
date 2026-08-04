"""
Diagnostic: shows which real AQICN station each city is actually being
matched to, and the distance. Helps us see if multiple cities are
collapsing onto the same station (which would explain duplicate AQI values).
"""
from cities import CITIES
from fetch_data import fetch_aqicn
import time

for city_info in CITIES:
    try:
        data = fetch_aqicn(city_info["lat"], city_info["lon"])
        station_name = data.get("city", {}).get("name", "?")
        station_geo = data.get("city", {}).get("geo", "?")
        print(f"{city_info['city']:20s} -> station: {station_name}  ({station_geo})")
    except Exception as e:
        print(f"{city_info['city']:20s} -> ERROR: {e}")
    time.sleep(1)
