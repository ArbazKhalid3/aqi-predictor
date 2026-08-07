"""
One-time cleanup: deletes the existing (partially contaminated) Hopsworks
feature group and recreates it fresh, then uploads the current clean local
CSV (1110 rows, post-cleanup). Safe to run once -- the hourly pipeline's
get_or_create_feature_group call will keep working unchanged afterward,
since it just finds this freshly-created group by the same name/version.
"""
import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

FEATURES_CSV = "../data/features.csv"

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
)
fs = project.get_feature_store()

# Delete the existing feature group (contains the 30 contaminated rows)
try:
    old_fg = fs.get_feature_group(name="aqi_features", version=1)
    old_fg.delete()
    print("Deleted old feature group.")
except Exception as e:
    print(f"No existing feature group to delete (or already gone): {e}")

# Load the clean local data and prep it the same way build_features.py does
df = pd.read_csv(FEATURES_CSV)
df["event_id"] = df["city"] + "_" + df["timestamp"]
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Recreate the feature group fresh -- same name/version/schema, so the
# hourly pipeline's get_or_create_feature_group finds this one going forward
fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    primary_key=["event_id"],
    event_time="timestamp",
    description="Hourly AQI + weather features per Pakistani city",
    time_travel_format="HUDI",
)
fg.insert(df, wait=True)
print(f"Uploaded {len(df)} clean rows to the fresh feature group.")
