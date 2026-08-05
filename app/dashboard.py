"""
Pearls AQI Predictor -- dashboard (Step 6, polished).

Shows the LATEST real reading per city (no forecast yet -- that gets
wired in once train_model.py has enough data to produce one). Now
includes a color-coded map of all 30 cities, a last-updated timestamp,
WHO/EPA guideline reference lines on the trend chart, and a page icon.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

FEATURES_CSV = "data/features.csv"

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")
st.title("🌫️ Pakistan AQI Predictor")
st.caption("10Pearls Internship Project — real-time AQI with 3-day forecasting")


@st.cache_data(ttl=300)  # refresh cache every 5 min, no need to reload on every click
def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_data()

# "Last updated" -- most recent timestamp across ALL cities, so the
# dashboard visibly proves it's live data, not a static snapshot.
last_updated = df["timestamp"].max()
st.caption(f"📡 Data last updated: {last_updated.strftime('%b %d, %Y — %I:%M %p UTC')}")


def aqi_category(aqi):
    """EPA AQI category bands -- used for labels, alerts, and map marker colors."""
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


# --- Latest reading per city, for the map ---
latest_per_city = (
    df.sort_values("timestamp").groupby("city").tail(1).reset_index(drop=True)
)
latest_per_city["category"], latest_per_city["color"] = zip(
    *latest_per_city["aqi"].map(aqi_category)
)

# --- Map: color-coded AQI across all cities ---
st.subheader("🗺️ AQI Across Pakistan")
fig_map = px.scatter_mapbox(
    latest_per_city,
    lat="lat", lon="lon",
    color="category",
    size=[20] * len(latest_per_city),  # uniform marker size, color carries the meaning
    hover_name="city",
    hover_data={"aqi": True, "lat": False, "lon": False, "category": False},
    color_discrete_map={
        "Good": "#00e400",
        "Moderate": "#ffff00",
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023",
    },
    zoom=4.3,
    center={"lat": 30.3753, "lon": 69.3451},  # Pakistan's approximate center
    mapbox_style="open-street-map",
    height=500,
)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)

# --- Country / City filter (per your multi-city requirement) ---
col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", sorted(df["country"].unique()))
with col2:
    cities_in_country = sorted(df[df["country"] == country]["city"].unique())
    city = st.selectbox("City", cities_in_country)

city_df = df[df["city"] == city].sort_values("timestamp")
latest = city_df.iloc[-1]
category, color = aqi_category(latest["aqi"])

# --- Current AQI display ---
st.subheader(f"{city}, {country}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current AQI", int(latest["aqi"]))
m2.metric("Category", category)
m3.metric("Temperature", f"{latest['temp_c']:.1f}°C")
m4.metric("Humidity", f"{latest['humidity']:.0f}%")

# Hazardous alert (per your brief's requirement)
if latest["aqi"] > 200:
    st.error(f"⚠️ Hazardous air quality alert for {city}! AQI = {int(latest['aqi'])} ({category})")
elif latest["aqi"] > 150:
    st.warning(f"⚠️ Unhealthy air quality in {city}. AQI = {int(latest['aqi'])} ({category})")

# --- Historical trend chart with WHO/EPA guideline reference lines ---
st.subheader("Recent AQI trend")
fig = px.line(city_df, x="timestamp", y="aqi", markers=True, title=f"AQI over time — {city}")

# EPA guideline bands, for context on how "bad" the current trend actually is
fig.add_hline(y=100, line_dash="dot", line_color="orange",
              annotation_text="Moderate threshold (AQI 100)", annotation_position="top left")
fig.add_hline(y=150, line_dash="dot", line_color="red",
              annotation_text="Unhealthy threshold (AQI 150)", annotation_position="top left")
fig.add_hline(y=300, line_dash="dot", line_color="purple",
              annotation_text="Hazardous threshold (AQI 300)", annotation_position="top left")

st.plotly_chart(fig, use_container_width=True)

# --- Placeholder for forecast (wired in once model.joblib exists) ---
st.subheader("3-Day Forecast")
st.info("Forecast will appear here once enough historical data has been "
        "collected to train the prediction model (currently accumulating).")

# --- Pollutant breakdown ---
st.subheader("Current pollutant levels")
pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
pollutant_data = latest[pollutant_cols].dropna()
if not pollutant_data.empty:
    # log scale: CO's naturally much larger µg/m³ values would otherwise
    # visually flatten PM2.5/PM10/NO2/SO2/O3 to near-zero on a linear axis
    fig2 = px.bar(x=pollutant_data.index, y=pollutant_data.values,
                  labels={"x": "Pollutant", "y": "Concentration (µg/m³, log scale)"},
                  log_y=True)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No pollutant data available for this reading.")
