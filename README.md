<div align="center">

# 🌫️ Pakistan AQI Predictor

### A Serverless, End-to-End Air Quality Forecasting System

[![Live Dashboard](https://img.shields.io/badge/🟢_Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge)](https://aqi-predictor-ydr7bnqs9d5twyutmwjhep.streamlit.app)
[![GitHub Actions](https://img.shields.io/badge/Automation-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/ArbazKhalid3/aqi-predictor/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

Predicts AQI **day-by-day for the next 3 days** across **30 Pakistani cities**, combining an automated hourly data pipeline, a cloud feature store, three separately trained forecasting models, and a fully automated retraining pipeline — all served through a live public dashboard.

**📄 [Full Project Report](docs/AQI_Predictor_Internship_Report.pdf)** · **🔗 [Live Dashboard](https://aqi-predictor-ydr7bnqs9d5twyutmwjhep.streamlit.app)** · **🧪 [GitHub Actions](https://github.com/ArbazKhalid3/aqi-predictor/actions)**

</div>

---

## 🛠️ Tech Stack

**Languages & Core**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

**Machine Learning**

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0092CA?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2?style=flat-square)

**Data & Visualization**

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

**Infrastructure & Automation**

![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![Hopsworks](https://img.shields.io/badge/Hopsworks-feature_store-1A73E8?style=flat-square)
![joblib](https://img.shields.io/badge/joblib-model_serialization-4B8BBE?style=flat-square)

---

## 📋 What It Does

- 📡 Collects hourly weather + air pollutant data for 30 Pakistani cities from OpenWeather's Air Pollution API
- 🧮 Computes standardized **US EPA AQI** from raw pollutant concentrations (PM2.5, PM10) — not a third-party AQI number
- ✅ Cross-validates against real **AQICN ground-station** readings for 4 cities (Karachi, Lahore, Islamabad, Peshawar), applying a per-city bias correction only where real validation data supports it
- 🗄️ Stores features in a **Hopsworks feature store**, plus a version-controlled local CSV
- 🤖 Trains **three separate models — one per forecast horizon** (24h / 48h / 72h) — comparing RandomForest, Ridge Regression, and XGBoost, automatically keeping the best performer per horizon
- 🧠 Also compared a TensorFlow feedforward neural network (Colab) during model selection — tree-based models won at current data volumes
- 🔍 Explains predictions with **SHAP** feature importance
- 📊 Serves everything through an interactive, publicly-accessible **Streamlit dashboard**

## 📱 Dashboard Pages

| Page | What it shows |
|---|---|
| 🏠 **Home** | National KPI summary, interactive map, city ranking (validated vs. estimate), AQI trend, 7-day heatmap, 3-day forecast, pollutant breakdown |
| ⚖️ **Compare** | Overlay AQI trends and pollutant levels for up to 4 cities side by side |
| 🩺 **Health** | Category-specific outdoor activity, sensitive-group, and mask guidance |
| 📈 **Performance** | R² / MAE comparison of all 3 production models, per forecast horizon |
| ℹ️ **About** | Project summary and developer info |
| 📋 **Methodology** | Full write-up of the data pipeline, calibration, models, and explainability |

## 🏗️ Architecture

```
OpenWeather API ─┐
                  ├──► Hourly Feature Pipeline (GitHub Actions)
   AQICN API   ───┘              │
                         ┌────────┴────────┐
                         ▼                 ▼
                    Hopsworks         features.csv
                  (feature store)    (GitHub repo)
                                          │
                                          ▼
                          Retraining Pipeline — every 4 days
                             (GitHub Actions, trains all 3 models)
                                          │
                                          ▼
                        model.joblib + model_metrics.csv
                                          │
                                          ▼
                              Streamlit Dashboard (live)
```

*(Full diagram with legend available in the project report.)*

## ✅ Proof of Automation

This isn't just described — it's independently visible in the repo. GitHub attributes every scheduled-workflow commit to a separate **`actions-user`** contributor, distinct from manual commits:

- **Hourly Feature Pipeline** — 400+ successful automated runs and counting
- **Retrain Models** — automated retraining every 4 days, each one committing an updated model + fresh metrics

Check the [Actions tab](https://github.com/ArbazKhalid3/aqi-predictor/actions) or the repo's [contributors](https://github.com/ArbazKhalid3/aqi-predictor/graphs/contributors) to see it live.

## 📂 Project Structure

```
scripts/build_features.py    — hourly data fetching + feature engineering
scripts/train_model.py       — trains RandomForest / Ridge / XGBoost per horizon
app/dashboard.py             — the Streamlit dashboard
notebooks/                   — EDA, calibration analysis, NN comparison
data/features.csv            — the feature store (hourly readings)
data/model.joblib            — current best model per horizon
data/model_metrics.csv       — R²/MAE/RMSE for every model at every horizon
docs/                        — full project report (PDF)
.github/workflows/hourly_features.yml   — automated hourly data collection
.github/workflows/retrain_model.yml     — automated model retraining
```

## ⚠️ Known Limitations

- Only 4 of 30 cities have real ground-truth validation data; the other 26 rely on uncorrected model estimates, clearly labeled as such on the dashboard
- The neural network comparison was a one-time Colab run and is not part of the automated retraining pipeline
- Model accuracy is expected to improve as more historical data accumulates over time

*(Full limitations list, methodology, and challenges faced are documented in the project report.)*

---

<div align="center">

**Muhammad Arbaz Khalid** — Data Science Student, Riphah International University
Built as a 10Pearls Internship Project (July 13 – September 4, 2026)

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ArbazKhalid3)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arbaz-khalid-bb9187279)

</div>
