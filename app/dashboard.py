"""
Pearls AQI Predictor -- dashboard with a real 3-DAY forecast (one
prediction per day, using three separately trained horizon models --
24h/48h/72h -- rather than a single distant number).
"""
import os
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

FEATURES_CSV = "data/features.csv"
MODEL_PATH = "data/model.joblib"

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")
st.title("🌫️ Pakistan AQI Predictor")
st.caption("10Pearls Internship Project — real-time AQI with 3-day forecasting")


@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


@st.cache_resource
def load_models():
    """Returns a dict keyed by horizon hours: {24: {...}, 48: {...}, 72: {...}}"""
    if not os.path.exists(MODEL_PATH):
        return {}
    return joblib.load(MODEL_PATH)


df = load_data()
horizon_models = load_models()

last_updated = df["timestamp"].max()
st.caption(f"📡 Data last updated: {last_updated.strftime('%b %d, %Y — %I:%M %p UTC')}")


def aqi_category(aqi):
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


latest_per_city = (
    df.sort_values("timestamp").groupby("city").tail(1).reset_index(drop=True)
)
latest_per_city["category"], latest_per_city["color"] = zip(
    *latest_per_city["aqi"].map(aqi_category)
)

st.subheader("🗺️ AQI Across Pakistan")
fig_map = px.scatter_mapbox(
    latest_per_city,
    lat="lat", lon="lon",
    color="category",
    size=[20] * len(latest_per_city),
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
    center={"lat": 30.3753, "lon": 69.3451},
    mapbox_style="open-street-map",
    height=500,
)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", sorted(df["country"].unique()))
with col2:
    cities_in_country = sorted(df[df["country"] == country]["city"].unique())
    city = st.selectbox("City", cities_in_country)

city_df = df[df["city"] == city].sort_values("timestamp")
latest = city_df.iloc[-1]
category, color = aqi_category(latest["aqi"])

st.subheader(f"{city}, {country}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current AQI", int(latest["aqi"]))
m2.metric("Category", category)
m3.metric("Temperature", f"{latest['temp_c']:.1f}°C")
m4.metric("Humidity", f"{latest['humidity']:.0f}%")

if latest["aqi"] > 200:
    st.error(f"⚠️ Hazardous air quality alert for {city}! AQI = {int(latest['aqi'])} ({category})")
elif latest["aqi"] > 150:
    st.warning(f"⚠️ Unhealthy air quality in {city}. AQI = {int(latest['aqi'])} ({category})")

st.subheader("Recent AQI trend")
fig = px.line(city_df, x="timestamp", y="aqi", markers=True, title=f"AQI over time — {city}")
fig.add_hline(y=100, line_dash="dot", line_color="orange",
              annotation_text="Moderate threshold (AQI 100)", annotation_position="top left")
fig.add_hline(y=150, line_dash="dot", line_color="red",
              annotation_text="Unhealthy threshold (AQI 150)", annotation_position="top left")
fig.add_hline(y=300, line_dash="dot", line_color="purple",
              annotation_text="Hazardous threshold (AQI 300)", annotation_position="top left")
st.plotly_chart(fig, use_container_width=True)

# --- Real 3-DAY forecast: one card per day, each from its own model ---
st.subheader("3-Day Forecast")
if not horizon_models:
    st.info("Forecast will appear here once enough historical data has been "
            "collected to train the prediction models (currently accumulating).")
else:
    features = latest[FEATURE_COLUMNS].fillna(0).to_frame().T
    day_labels = {24: "Day 1 (Tomorrow)", 48: "Day 2", 72: "Day 3"}
    day_dates = {h: (last_updated + pd.Timedelta(hours=h)).strftime("%b %d") for h in [24, 48, 72]}

    cols = st.columns(3)
    for i, horizon in enumerate([24, 48, 72]):
        with cols[i]:
            if horizon not in horizon_models:
                st.info(f"{day_labels[horizon]}\nNot enough data yet for this horizon.")
                continue

            model_info = horizon_models[horizon]
            predicted_aqi = model_info["model"].predict(features)[0]
            pred_category, pred_color = aqi_category(predicted_aqi)

            st.markdown(f"**{day_labels[horizon]}** — {day_dates[horizon]}")
            st.metric("Predicted AQI", f"{predicted_aqi:.0f}",
                       delta=f"{predicted_aqi - latest['aqi']:.0f} vs today")
            st.write(pred_category)
            st.caption(f"Model: {model_info['name']}")

            if predicted_aqi > 200:
                st.error("⚠️ Hazardous")
            elif predicted_aqi > 150:
                st.warning("⚠️ Unhealthy")

    st.caption("Each day uses a separately trained model for that specific horizon "
               "(24h/48h/72h). Accuracy improves as more historical data is collected.")

st.subheader("Current pollutant levels")
pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
pollutant_data = latest[pollutant_cols].dropna()
if not pollutant_data.empty:
    fig2 = px.bar(x=pollutant_data.index, y=pollutant_data.values,
                  labels={"x": "Pollutant", "y": "Concentration (µg/m³, log scale)"},
                  log_y=True)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No pollutant data available for this reading.")
