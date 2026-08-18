import pandas as pd

df = pd.read_csv("../data/features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

for city in ["Karachi", "Lahore", "Islamabad", "Peshawar"]:
    city_df = df[df["city"] == city].sort_values("timestamp")
    after = city_df[city_df["timestamp"] >= "2026-08-11"]
    clamped_pct = (after["aqi"] == 0).mean() * 100
    print(f"{city}: {len(after)} rows, {clamped_pct:.1f}% clamped to 0, "
          f"mean={after['aqi'].mean():.1f}, median={after['aqi'].median():.1f}")
