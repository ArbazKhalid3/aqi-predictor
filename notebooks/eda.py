# %% [markdown]
# # AQI Predictor — Exploratory Data Analysis
# Pakistan AQI Predictor — 10Pearls Internship Project

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
pd.set_option("display.max_columns", None)

df = pd.read_csv("../data/features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
print(f"Loaded {len(df)} rows, {df['city'].nunique()} cities")
df.head()

# %%
# Basic shape and data quality overview
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("\nRows per city:")
print(df["city"].value_counts())

print("\nMissing values per column:")
print(df.isnull().sum())

# %%
# Summary statistics for AQI and key pollutants
df[["aqi", "pm25", "pm10", "co", "no2", "so2", "o3", "temp_c", "humidity"]].describe()

# %%
# AQI trend over time for a few representative cities
sample_cities = ["Karachi", "Lahore", "Islamabad", "Quetta", "Rahim Yar Khan"]
plt.figure(figsize=(14, 6))
for city in sample_cities:
    city_data = df[df["city"] == city].sort_values("timestamp")
    plt.plot(city_data["timestamp"], city_data["aqi"], marker="o", markersize=3, label=city)
plt.axhline(y=150, color="red", linestyle="--", alpha=0.5, label="Unhealthy threshold")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("AQI Over Time — Selected Cities")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots_aqi_trend.png", dpi=100)
plt.show()

# %%
# Average AQI by city, ranked worst to best -- gives a first sense of
# which cities are consistently more polluted vs which just spike occasionally
city_avg_aqi = df.groupby("city")["aqi"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 10))
city_avg_aqi.plot(kind="barh", color="steelblue")
plt.xlabel("Average AQI")
plt.title("Average AQI by City (across all collected hours so far)")
plt.tight_layout()
plt.savefig("plots_city_avg_aqi.png", dpi=100)
plt.show()

print(city_avg_aqi)

# %%
# Correlation heatmap: which features actually relate to AQI?
# Useful both for understanding the data and for justifying feature choices later.
corr_cols = ["aqi", "pm25", "pm10", "co", "no2", "so2", "o3",
             "temp_c", "humidity", "pressure", "wind_speed"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots_correlation.png", dpi=100)
plt.show()

# %%
# AQI by hour of day -- averaged across all cities, shows daily pollution rhythm
plt.figure(figsize=(10, 5))
hourly_avg = df.groupby("hour")["aqi"].mean()
hourly_avg.plot(kind="line", marker="o", color="darkorange")
plt.xlabel("Hour of day (UTC)")
plt.ylabel("Average AQI")
plt.title("Average AQI by Hour of Day (all cities combined)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots_hourly_pattern.png", dpi=100)
plt.show()

# %%
# AQI category breakdown -- what proportion of all readings fall into each
# EPA band. Directly relevant to the "hazardous alert" feature in the dashboard.
def aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

df["category"] = df["aqi"].apply(aqi_category)
category_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                   "Unhealthy", "Very Unhealthy", "Hazardous"]
category_counts = df["category"].value_counts().reindex(category_order).fillna(0)

plt.figure(figsize=(10, 6))
colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
category_counts.plot(kind="bar", color=colors)
plt.xlabel("AQI Category")
plt.ylabel("Number of readings")
plt.title("Distribution of AQI Readings by Category (all cities, all time)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plots_category_distribution.png", dpi=100)
plt.show()

print(category_counts)
print(f"\n% of readings Unhealthy or worse: {(df['aqi'] > 150).mean() * 100:.1f}%")

# %%
# AQI spread per city (boxplot) -- shows which cities have wide variability
# (like Quetta's spikes) vs consistently stable readings (like Karachi)
plt.figure(figsize=(16, 8))
city_order = df.groupby("city")["aqi"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="city", y="aqi", order=city_order)
plt.axhline(y=150, color="red", linestyle="--", alpha=0.5)
plt.xticks(rotation=90)
plt.ylabel("AQI")
plt.title("AQI Distribution by City (sorted by median)")
plt.tight_layout()
plt.savefig("plots_city_boxplot.png", dpi=100)
plt.show()

# %%
# Investigate the AQI=500 spikes: are they clustered in the same hourly
# run across many cities (suggesting a pipeline/API glitch), or spread
# out independently per city (suggesting genuine separate events)?
spikes = df[df["aqi"] >= 490].sort_values("timestamp")
print(f"Total readings at/near the 500 cap: {len(spikes)}")
print(spikes[["city", "timestamp", "aqi", "pm25", "pm10"]].to_string())

print("\nSpikes grouped by rounded hour (are many cities spiking at the SAME time?):")
spikes_copy = spikes.copy()
spikes_copy["hour_bucket"] = spikes_copy["timestamp"].dt.floor("h")
print(spikes_copy.groupby("hour_bucket")["city"].apply(list))

# %%
# Weather vs AQI relationships -- do temperature, humidity, and wind
# actually show visible patterns with AQI, or is the correlation too
# weak to matter? Scatter plots make this easier to judge than raw
# correlation numbers alone.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(df["temp_c"], df["aqi"], alpha=0.3, s=15)
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("AQI")
axes[0].set_title("Temperature vs AQI")

axes[1].scatter(df["humidity"], df["aqi"], alpha=0.3, s=15, color="steelblue")
axes[1].set_xlabel("Humidity (%)")
axes[1].set_ylabel("AQI")
axes[1].set_title("Humidity vs AQI")

axes[2].scatter(df["wind_speed"], df["aqi"], alpha=0.3, s=15, color="seagreen")
axes[2].set_xlabel("Wind Speed (m/s)")
axes[2].set_ylabel("AQI")
axes[2].set_title("Wind Speed vs AQI")

plt.tight_layout()
plt.savefig("plots_weather_scatter.png", dpi=100)
plt.show()

# %%
# Day-of-week pattern -- do weekdays (more traffic/industry) show higher
# AQI than weekends? Relevant given Pakistan's Friday-Saturday or
# Saturday-Sunday weekend patterns vary by sector.
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_avg = df.groupby("day_of_week")["aqi"].mean()
plt.figure(figsize=(8, 5))
plt.bar([day_names[i] for i in dow_avg.index], dow_avg.values, color="coral")
plt.ylabel("Average AQI")
plt.title("Average AQI by Day of Week")
plt.tight_layout()
plt.savefig("plots_day_of_week.png", dpi=100)
plt.show()
print(dow_avg)
