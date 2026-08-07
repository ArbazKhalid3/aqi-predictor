"""
Retry the Hopsworks resync in small batches (30 rows at a time, matching
the size that succeeds every hour in the normal pipeline) instead of one
big 1110-row upload, which failed -- likely a free-tier resource limit on
large materialization jobs.
"""
import os
import time
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

FEATURES_CSV = "../data/features.csv"
BATCH_SIZE = 30

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
)
fs = project.get_feature_store()

# The empty feature group already exists from the previous run -- just
# fetch it rather than recreating (get_or_create finds it either way)
fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    primary_key=["event_id"],
    event_time="timestamp",
    description="Hourly AQI + weather features per Pakistani city",
    time_travel_format="HUDI",
)

df = pd.read_csv(FEATURES_CSV)
df["event_id"] = df["city"] + "_" + df["timestamp"]
df["timestamp"] = pd.to_datetime(df["timestamp"])

total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Uploading {len(df)} rows in {total_batches} batches of {BATCH_SIZE}...")

for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    try:
        fg.insert(batch, wait=False)  # fire-and-forget, same as the hourly pipeline
        print(f"  Batch {batch_num}/{total_batches}: submitted {len(batch)} rows")
    except Exception as e:
        print(f"  Batch {batch_num}/{total_batches}: FAILED - {e}")
    time.sleep(3)  # brief pause between batches so we don't overload the same job queue

print("\nAll batches submitted. Check the Hopsworks UI in a few minutes to confirm row count.")
