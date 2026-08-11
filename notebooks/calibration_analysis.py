# %% [markdown]
# # Calibration Analysis: Computed AQI vs Real AQICN Station Readings
# Checks how well our OpenWeather-derived computed AQI tracks real
# ground-station AQI for the 4 cities with genuine AQICN coverage
# (Karachi, Lahore, Islamabad, Peshawar).

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_theme(style="darkgrid")

df = pd.read_csv("../data/features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

# Only rows where we actually have a real station reading to compare against
calib_df = df.dropna(subset=["aqi_reference"]).copy()
print(f"{len(calib_df)} rows with a real AQICN reference reading")
print(calib_df["city"].value_counts())

# %%
# Scatter: computed AQI (x) vs real station AQI (y), per city.
# A perfect model would fall exactly on the diagonal line.
plt.figure(figsize=(9, 9))
for city in calib_df["city"].unique():
    city_data = calib_df[calib_df["city"] == city]
    plt.scatter(city_data["aqi"], city_data["aqi_reference"], label=city, alpha=0.6, s=40)

max_val = max(calib_df["aqi"].max(), calib_df["aqi_reference"].max())
plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect agreement")
plt.xlabel("Computed AQI (OpenWeather-derived)")
plt.ylabel("Real AQICN Station AQI")
plt.title("Computed vs Real AQI, by City")
plt.legend()
plt.tight_layout()
plt.savefig("plots_calibration_scatter.png", dpi=100)
plt.show()

# %%
# Per-city bias: on average, does our computed AQI run high or low
# relative to the real station, and by how much?
calib_df["error"] = calib_df["aqi"] - calib_df["aqi_reference"]
bias_by_city = calib_df.groupby("city")["error"].agg(["mean", "std", "count"])
print("Bias per city (computed - real):")
print(bias_by_city)

mae = mean_absolute_error(calib_df["aqi_reference"], calib_df["aqi"])
print(f"\nOverall MAE (computed vs real): {mae:.1f} AQI points")

# %%
# Try fitting a single correction: real_aqi ≈ a * computed_aqi + b
# If this fits reasonably well (decent R²), it means a simple linear
# calibration could meaningfully improve accuracy for ALL 30 cities,
# not just the 4 with ground truth.
X = calib_df[["aqi"]].values
y = calib_df["aqi_reference"].values
reg = LinearRegression().fit(X, y)
r2 = r2_score(y, reg.predict(X))

print(f"Linear correction: real_aqi ≈ {reg.coef_[0]:.3f} * computed_aqi + {reg.intercept_:.1f}")
print(f"R² of this correction: {r2:.3f}")
