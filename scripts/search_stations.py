"""
Diagnostic: searches AQICN by city name (not coordinates) to find real
stations, and checks whether the match is actually in Pakistan.
This avoids the geo: endpoint's bad global fallback behavior.
"""
import requests
import os
import time
from dotenv import load_dotenv
from cities import CITIES

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

for city_info in CITIES:
    city = city_info["city"]
    url = "https://api.waqi.info/search/"
    resp = requests.get(url, params={"keyword": city, "token": TOKEN}, timeout=10).json()

    if resp.get("status") != "ok" or not resp.get("data"):
        print(f"{city:20s} -> NO MATCH")
        continue

    # Print top result's station name and its country hint
    top = resp["data"][0]
    station_name = top["station"]["name"]
    uid = top["uid"]
    print(f"{city:20s} -> [{uid}] {station_name}")
    time.sleep(1)
