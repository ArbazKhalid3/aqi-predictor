"""
Pearls AQI Predictor -- Home / Compare Cities / Health Advice /
Model Performance / About / Methodology, with KPI row, legend above
the map, validated-city badges, 7-day heatmap, 3-day forecast, and a
consistent footer across all pages.
"""
import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

FEATURES_CSV = "data/features.csv"
MODEL_PATH = "data/model.joblib"
METRICS_PATH = "data/model_metrics.csv"  # optional: columns = horizon,model,r2,mae

VALIDATED_CITIES = {"Karachi", "Lahore", "Islamabad", "Peshawar"}

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid rgba(255,255,255,0.1);
}
.stTabs [data-baseweb="tab"] {
    height: 56px !important;
    padding: 0 26px !important;
    border-radius: 10px 10px 0 0;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div,
.stTabs [data-baseweb="tab"] span {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(100,150,255,0.15) !important;
    border-bottom: 4px solid #4f8cff !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div,
.stTabs [aria-selected="true"] span {
    color: #6fa8ff !important;
    font-weight: 800 !important;
}
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.badge {
    display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
    border-radius: 20px; background: rgba(100,150,255,0.15);
    border: 1px solid rgba(100,150,255,0.35);
    font-size: 0.82em;
}
.kpi-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px 18px; text-align: center;
}
.validated-tag {
    font-size: 0.72em; padding: 1px 8px; border-radius: 10px;
    background: rgba(0,200,100,0.15); border: 1px solid rgba(0,200,100,0.4);
    color: #4dd68a;
}
.estimated-tag {
    font-size: 0.72em; padding: 1px 8px; border-radius: 10px;
    background: rgba(255,180,0,0.12); border: 1px solid rgba(255,180,0,0.35);
    color: #ffb800;
}
.app-footer {
    margin-top: 48px; padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 0.78em; opacity: 0.55; text-align: center;
}
.app-footer a { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return {}
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=300)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)


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


CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}

HEALTH_ADVICE = {
    "Good": {
        "icon": "🟢",
        "summary": "Air quality is satisfactory. Enjoy normal outdoor activities.",
        "outdoor": "No restrictions — safe for all outdoor activity, including exercise.",
        "sensitive": "No special precautions needed.",
        "mask": "Not required.",
    },
    "Moderate": {
        "icon": "🟡",
        "summary": "Air quality is acceptable, though a small risk exists for unusually sensitive individuals.",
        "outdoor": "Generally safe. Unusually sensitive people should watch for symptoms during prolonged exertion.",
        "sensitive": "People with respiratory conditions (asthma etc.) should consider reducing prolonged outdoor exertion.",
        "mask": "Optional for sensitive individuals.",
    },
    "Unhealthy for Sensitive Groups": {
        "icon": "🟠",
        "summary": "Sensitive groups may experience health effects; the general public is less likely to be affected.",
        "outdoor": "Reduce prolonged or heavy outdoor exertion, especially children, elderly, and those with heart/lung conditions.",
        "sensitive": "Children, elderly, pregnant women, and people with asthma/heart disease should limit outdoor time.",
        "mask": "Recommended for sensitive groups when outdoors for extended periods.",
    },
    "Unhealthy": {
        "icon": "🔴",
        "summary": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
        "outdoor": "Avoid prolonged outdoor exertion. Keep outdoor activity short for everyone.",
        "sensitive": "Sensitive groups should avoid outdoor activity entirely where possible.",
        "mask": "N95/KN95 mask recommended when outdoors.",
    },
    "Very Unhealthy": {
        "icon": "🟣",
        "summary": "Health alert: everyone may experience more serious health effects.",
        "outdoor": "Avoid all outdoor exertion. Stay indoors with windows closed if possible.",
        "sensitive": "Sensitive groups should remain indoors at all times.",
        "mask": "N95/KN95 mask required if going outside is unavoidable.",
    },
    "Hazardous": {
        "icon": "⚫",
        "summary": "Health warning of emergency conditions — the entire population is at risk.",
        "outdoor": "Stay indoors. Avoid all outdoor activity.",
        "sensitive": "Sensitive groups face serious risk even briefly outdoors — remain indoors with air filtration if available.",
        "mask": "N95/KN95 mask essential; consider an air purifier indoors.",
    },
}


def render_footer():
    st.markdown(
        "<div class='app-footer'>"
        "🌫️ Pakistan AQI Predictor &nbsp;·&nbsp; Built by Muhammad Arbaz Khalid for the 10Pearls Internship "
        "&nbsp;·&nbsp; Data refreshed hourly via GitHub Actions "
        "&nbsp;·&nbsp; <a href='https://github.com/ArbazKhalid3/aqi-predictor' target='_blank'>Source on GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_legend():
    st.caption("**AQI Category Legend**")
    cols = st.columns(6)
    for i, (label, color) in enumerate(CATEGORY_COLORS.items()):
        with cols[i]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;font-size:0.8em;'>"
                f"<div style='width:12px;height:12px;background:{color};border-radius:50%;'></div>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )


def render_kpi_row(latest_per_city):
    avg_aqi = latest_per_city["aqi"].mean()
    best = latest_per_city.loc[latest_per_city["aqi"].idxmin()]
    worst = latest_per_city.loc[latest_per_city["aqi"].idxmax()]
    unhealthy_count = (latest_per_city["aqi"] > 150).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-box'><div style='font-size:0.8em;opacity:0.7'>National Average AQI</div>"
                     f"<div style='font-size:1.8em;font-weight:700'>{avg_aqi:.0f}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-box'><div style='font-size:0.8em;opacity:0.7'>Best Air Quality</div>"
                     f"<div style='font-size:1.4em;font-weight:700'>{best['city']}</div>"
                     f"<div style='font-size:0.85em;opacity:0.8'>AQI {int(best['aqi'])}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-box'><div style='font-size:0.8em;opacity:0.7'>Worst Air Quality</div>"
                     f"<div style='font-size:1.4em;font-weight:700'>{worst['city']}</div>"
                     f"<div style='font-size:0.85em;opacity:0.8'>AQI {int(worst['aqi'])}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-box'><div style='font-size:0.8em;opacity:0.7'>Cities Above Unhealthy (150)</div>"
                     f"<div style='font-size:1.8em;font-weight:700'>{unhealthy_count} / {len(latest_per_city)}</div></div>", unsafe_allow_html=True)


def render_heatmap(city_df, city):
    recent = city_df.copy()
    recent["date"] = recent["timestamp"].dt.strftime("%a %b %d")
    recent["hour_of_day"] = recent["timestamp"].dt.hour

    last_7_dates = recent["date"].drop_duplicates().tail(7).tolist()
    recent = recent[recent["date"].isin(last_7_dates)]

    pivot = recent.pivot_table(index="date", columns="hour_of_day", values="aqi", aggfunc="mean")
    pivot = pivot.reindex(last_7_dates)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0.0, "#00e400"], [0.2, "#ffff00"], [0.4, "#ff7e00"],
            [0.6, "#ff0000"], [0.8, "#8f3f97"], [1.0, "#7e0023"],
        ],
        zmin=0, zmax=300,
        colorbar=dict(title="AQI"),
        hovertemplate="Date: %{y}<br>Hour: %{x}<br>AQI: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(title=f"7-Day AQI Heatmap — {city}", height=350,
                       xaxis_title="Hour of day (UTC)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_map(latest_per_city):
    st.subheader("🗺️ AQI Across Pakistan")
    render_legend()
    fig_map = px.scatter_mapbox(
        latest_per_city,
        lat="lat", lon="lon",
        color="category",
        size=[20] * len(latest_per_city),
        text=latest_per_city["aqi"].astype(int).astype(str),
        hover_name="city",
        hover_data={"aqi": True, "lat": False, "lon": False, "category": False},
        color_discrete_map=CATEGORY_COLORS,
        zoom=4.3,
        center={"lat": 30.3753, "lon": 69.3451},
        mapbox_style="open-street-map",
        height=500,
    )
    fig_map.update_traces(textfont=dict(size=10, color="black"), textposition="middle center")
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("🟢 Ground-truth validated: Karachi, Lahore, Islamabad, Peshawar. Other cities use uncorrected satellite/model estimates.")


def render_home(df, horizon_models):
    st.title("🏠 Pakistan AQI Predictor")
    st.caption("10Pearls Internship Project — real-time AQI with 3-day forecasting")

    last_updated = df["timestamp"].max()
    st.caption(f"📡 Data last updated: {last_updated.strftime('%b %d, %Y — %I:%M %p UTC')}")
    st.divider()

    latest_per_city = (
        df.sort_values("timestamp").groupby("city").tail(1).reset_index(drop=True)
    )
    latest_per_city["category"], latest_per_city["color"] = zip(
        *latest_per_city["aqi"].map(aqi_category)
    )

    render_kpi_row(latest_per_city)
    st.divider()

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
    tag_html = "<span class='validated-tag'>✓ Ground-truth validated</span>" if city in VALIDATED_CITIES \
        else "<span class='estimated-tag'>~ Model estimate (uncorrected)</span>"
    st.markdown(tag_html, unsafe_allow_html=True)

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

    st.subheader("📅 7-Day Heatmap")
    render_heatmap(city_df, city)

    st.subheader("3-Day Forecast")
    if not horizon_models:
        st.info("Forecast will appear here once enough historical data has been "
                "collected to train the prediction models (currently accumulating).")
    else:
        features = latest[FEATURE_COLUMNS].fillna(0).astype(float).to_frame().T
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

    st.divider()

    st.subheader("🏆 City AQI Ranking")
    ranking_df = latest_per_city[["city", "aqi", "category"]].sort_values("aqi", ascending=False).reset_index(drop=True)
    ranking_df.index = ranking_df.index + 1
    ranking_df["Source"] = ranking_df["city"].apply(lambda c: "✓ Validated" if c in VALIDATED_CITIES else "~ Estimate")
    ranking_df.columns = ["City", "AQI", "Category", "Source"]
    st.dataframe(ranking_df, use_container_width=True, height=350)

    st.divider()

    render_map(latest_per_city)
    render_footer()


def render_compare(df):
    st.title("⚖️ Compare Cities")
    st.caption("Overlay AQI trends for up to 4 cities side by side.")

    all_cities = sorted(df["city"].unique())
    default_cities = [c for c in ["Karachi", "Lahore", "Islamabad"] if c in all_cities] or all_cities[:2]
    selected = st.multiselect("Choose cities to compare", all_cities, default=default_cities, max_selections=4)

    if not selected:
        st.info("Pick at least one city to see the comparison.")
        render_footer()
        return

    compare_df = df[df["city"].isin(selected)].sort_values("timestamp")

    st.subheader("AQI trend comparison")
    fig = px.line(compare_df, x="timestamp", y="aqi", color="city", markers=True)
    fig.add_hline(y=100, line_dash="dot", line_color="orange")
    fig.add_hline(y=150, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current snapshot")
    latest = compare_df.groupby("city").tail(1).reset_index(drop=True)
    latest["category"], _ = zip(*latest["aqi"].map(aqi_category))
    latest["Source"] = latest["city"].apply(lambda c: "✓ Validated" if c in VALIDATED_CITIES else "~ Estimate")
    snap = latest[["city", "aqi", "category", "temp_c", "humidity", "Source"]].rename(
        columns={"city": "City", "aqi": "AQI", "category": "Category", "temp_c": "Temp (°C)", "humidity": "Humidity (%)"}
    )
    snap = snap.sort_values("AQI").reset_index(drop=True)
    snap.index = snap.index + 1
    st.dataframe(snap, use_container_width=True)

    st.subheader("Current pollutant breakdown")
    pollutant_cols = ["pm25", "pm10", "co", "no2", "so2", "o3"]
    melt = latest.melt(id_vars="city", value_vars=pollutant_cols, var_name="pollutant", value_name="value")
    fig2 = px.bar(melt, x="pollutant", y="value", color="city", barmode="group", log_y=True,
                  labels={"value": "Concentration (µg/m³, log scale)", "pollutant": "Pollutant"})
    st.plotly_chart(fig2, use_container_width=True)

    render_footer()


def render_health_advice(df):
    st.title("⚕️ Health Advice")
    st.caption("What today's AQI category means for you — based on your selected city.")

    all_cities = sorted(df["city"].unique())
    default_idx = all_cities.index("Karachi") if "Karachi" in all_cities else 0
    city = st.selectbox("City", all_cities, index=default_idx)

    city_df = df[df["city"] == city].sort_values("timestamp")
    latest = city_df.iloc[-1]
    category, color = aqi_category(latest["aqi"])
    advice = HEALTH_ADVICE[category]

    st.markdown(
        f"<div class='card' style='border-left:6px solid {color}'>"
        f"<div style='font-size:2.2em'>{advice['icon']} <b>{category}</b></div>"
        f"<div style='font-size:1.3em;margin-top:6px'>Current AQI in {city}: <b>{int(latest['aqi'])}</b></div>"
        f"<div style='margin-top:10px;opacity:0.85'>{advice['summary']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card'>🏃 <b>Outdoor Activity</b><br><br>{advice['outdoor']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'>👶 <b>Sensitive Groups</b><br><br>{advice['sensitive']}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'>😷 <b>Mask Guidance</b><br><br>{advice['mask']}</div>", unsafe_allow_html=True)

    st.caption("General guidance based on standard EPA AQI category recommendations — not a substitute for medical advice.")

    st.divider()
    st.subheader("All categories at a glance")
    for cat, info in HEALTH_ADVICE.items():
        with st.expander(f"{info['icon']} {cat}"):
            st.write(info["summary"])

    render_footer()


def render_model_performance():
    st.title("📈 Model Performance")
    st.caption("Comparison of the 4 forecasting approaches evaluated for this project.")

    metrics_df = load_metrics()

    if metrics_df is None:
        st.info("Model metrics file not found yet (data/model_metrics.csv). "
                "Once you export R\u00b2/MAE per model from your training notebook, this page will "
                "render comparison charts automatically.")
        st.markdown("**Expected columns for `data/model_metrics.csv`:** `horizon, model, r2, mae`")
        example = pd.DataFrame({
            "horizon": [24, 24, 24, 24],
            "model": ["Ridge Regression", "RandomForest", "XGBoost", "Neural Network"],
            "r2": [0.61, 0.84, 0.86, 0.79],
            "mae": [14.2, 8.1, 7.6, 9.4],
        })
        st.dataframe(example, use_container_width=True)
        st.caption("Add one row per model per horizon (24/48/72), save as CSV at that path, and reload this page.")
        render_footer()
        return

    horizons = sorted(metrics_df["horizon"].unique())
    tabs = st.tabs([f"{h}h horizon" for h in horizons])
    for tab, horizon in zip(tabs, horizons):
        with tab:
            sub = metrics_df[metrics_df["horizon"] == horizon].sort_values("r2", ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                fig_r2 = px.bar(sub, x="model", y="r2", title="R\u00b2 (higher is better)", range_y=[0, 1])
                st.plotly_chart(fig_r2, use_container_width=True)
            with c2:
                fig_mae = px.bar(sub, x="model", y="mae", title="MAE (lower is better)")
                st.plotly_chart(fig_mae, use_container_width=True)
            st.dataframe(sub.reset_index(drop=True), use_container_width=True)

    render_footer()


def render_about():
    st.title("ℹ️ About")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
### Pakistan AQI Predictor

A serverless, end-to-end air quality forecasting system covering **30 Pakistani
cities**, predicting AQI up to **3 days ahead** using ground-truth-calibrated
machine learning models.
""")
    st.markdown(
        '<span class="badge">🌫️ 30 Cities</span>'
        '<span class="badge">📈 3-Day Forecast</span>'
        '<span class="badge">🤖 4 ML Models Compared</span>'
        '<span class="badge">🔍 SHAP Explainability</span>'
        '<span class="badge">☁️ 100% Serverless</span>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            "<div style='width:80px;height:80px;border-radius:50%;background:linear-gradient(135deg,#4f8cff,#8f3f97);"
            "display:flex;align-items:center;justify-content:center;font-size:2em;'>👤</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("""
**Muhammad Arbaz Khalid**
Data Science Student, Riphah International University
Built as a 10Pearls Internship Project

🔗 [GitHub](https://github.com/ArbazKhalid3) · 🔗 [LinkedIn](https://www.linkedin.com/in/arbaz-khalid-bb9187279) · 📧 arbazkhalid653@gmail.com
""")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">💻 <b>Source Code</b><br><a href="https://github.com/ArbazKhalid3/aqi-predictor" target="_blank">github.com/ArbazKhalid3/aqi-predictor</a></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">🌐 <b>Live Dashboard</b><br><a href="https://aqi-predictor-ydr7bnqs9d5twyutmwjhep.streamlit.app" target="_blank">aqi-predictor-ydr7bnqs9d5twyutmwjhep.streamlit.app</a></div>', unsafe_allow_html=True)

    st.caption("💡 Tip: switch light/dark theme via menu (☰ top-right) → Settings → Choose app theme.")

    render_footer()


def render_methodology():
    st.title("📋 Methodology")
    st.caption("How the data is collected, calibrated, and turned into a 3-day forecast.")

    with st.expander("📡 Data Pipeline", expanded=True):
        st.markdown("""
Weather and pollutant data is collected **hourly** for 30 Pakistani cities via
GitHub Actions, using **OpenWeather's Air Pollution API** as the primary source
(atmospheric-model-based, works for any coordinates) and **AQICN ground-station**
readings for 4 cities (Karachi, Lahore, Islamabad, Peshawar) as validation.
Data is stored in both a **Hopsworks feature store** and a local CSV.
""")

    with st.expander("🧮 AQI Calculation"):
        st.markdown("""
AQI is computed from PM2.5/PM10 concentrations using the **official US EPA
breakpoint formula** (not a third-party AQI number), with proper value
truncation per EPA methodology.
""")

    with st.expander("⚖️ Calibration"):
        st.markdown("""
OpenWeather's underlying atmospheric model showed large, **city-specific**
bias against real ground-truth stations when validated (up to **~2.3x**
under/over-estimation). A single global correction did not generalize
(R² = 0.14 across all 4 cities).

A **per-city ratio-based correction** was applied instead, only for the 4
cities with real validation data — the other 26 cities remain uncorrected,
since applying an unvalidated correction risks making accuracy worse, not
better.
""")

    with st.expander("🤖 Models"):
        st.markdown("""
Four forecasting approaches were compared:

| Model | Type |
|---|---|
| Ridge Regression | Linear baseline |
| RandomForest | Tree-based |
| XGBoost | Tree-based |
| Feedforward Neural Network | Deep learning (TensorFlow, trained on Google Colab) |

Tree-based models (RandomForest/XGBoost) consistently outperformed both the
linear baseline and the neural network at current data volumes — an
expected pattern, since neural networks typically need larger datasets to
become competitive on tabular data.

The dashboard's 3-day forecast uses **three separately trained models**,
one per horizon (24h/48h/72h) — not a single model reused three times.
""")

    with st.expander("🔍 Explainability"):
        st.markdown("""
**SHAP** (SHapley Additive exPlanations) was used to identify which
features most influence predictions: **PM2.5** and **CO** consistently
ranked highest, with CO acting as a proxy for sustained local pollution
intensity even though it isn't a direct input to the AQI formula itself.
""")

    render_footer()


df = load_data()
horizon_models = load_models()

tab_home, tab_compare, tab_health, tab_perf, tab_about, tab_method = st.tabs(
    ["🏠 Home", "⚖️ Compare", "⚕️ Health", "📈 Performance", "ℹ️ About", "📋 Methodology"]
)

with tab_home:
    render_home(df, horizon_models)
with tab_compare:
    render_compare(df)
with tab_health:
    render_health_advice(df)
with tab_perf:
    render_model_performance()
with tab_about:
    render_about()
with tab_method:
    render_methodology()
