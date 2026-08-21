# Pakistan AQI Predictor

A serverless, end-to-end air quality forecasting system built for the 10Pearls internship project brief. Predicts AQI day-by-day for the next 3 days across 30 Pakistani cities using an automated hourly data pipeline, a Hopsworks feature store, three separately trained forecasting models, and an automated retraining pipeline — all served through a live dashboard.

**Live dashboard:** https://aqi-predictor-ydr7bnqs9d5twyutmwjhep.streamlit.app

## What it does

- Collects hourly weather + air pollutant data for 30 Pakistani cities from OpenWeather's Air Pollution API
- Computes standardized US EPA AQI from raw pollutant concentrations (PM2.5, PM10) — not a third-party AQI number
- Cross-validates against real AQICN ground-station readings for 4 cities (Karachi, Lahore, Islamabad, Peshawar), and applies a per-city ratio-based bias correction only where real validation data supports it — the other 26 cities are left uncorrected rather than guessed at
- Stores features in a Hopsworks feature store (in addition to a local CSV)
- Trains **three separate models — one per forecast horizon (24h / 48h / 72h)** — comparing RandomForest, Ridge Regression, and XGBoost at each horizon and automatically keeping the best performer per horizon
- Also compared a TensorFlow feedforward neural network (trained on Colab) during model selection; tree-based models won at current data volumes, so the NN was not carried into production
- Explains model predictions with SHAP feature importance
- Displays live readings, historical trends, day-by-day 3-day forecasts, city comparisons, health guidance, and model performance metrics on an interactive dashboard

## Dashboard pages

- **Home** — national KPI summary, interactive city map, city ranking (flagged validated vs. model-estimate), AQI trend, 7-day heatmap, 3-day forecast, pollutant breakdown
- **Compare** — overlay AQI trends and pollutant levels for up to 4 cities side by side
- **Health** — category-specific outdoor activity, sensitive-group, and mask guidance based on current AQI
- **Performance** — R² / MAE comparison of all 3 production models, per forecast horizon
- **About** — project summary and developer info
- **Methodology** — full write-up of the data pipeline, AQI calculation, calibration approach, model comparison, and explainability

## Architecture

OpenWeather API + AQICN API feed an **hourly feature pipeline** (automated via GitHub Actions), which writes to both a Hopsworks feature store and a local CSV. A separate **retraining pipeline** (automated via GitHub Actions, every 4 days) reads that same feature data, retrains all 3 horizon models, evaluates every model at every horizon, and commits both the updated model file and a metrics CSV. The live Streamlit dashboard reads the latest data, models, and metrics directly from the repo — so the whole system stays current with zero manual intervention between the two scheduled pipelines.

## Project structure

- `scripts/build_features.py` — hourly data fetching + feature engineering
- `scripts/train_model.py` — trains RandomForest / Ridge / XGBoost per horizon, saves the best model per horizon plus a full metrics comparison
- `app/dashboard.py` — the Streamlit dashboard
- `notebooks/` — EDA, calibration analysis, and neural network comparison notebooks
- `data/features.csv` — the feature store (hourly readings)
- `data/model.joblib` — the current best model per horizon
- `data/model_metrics.csv` — R²/MAE/RMSE for every model at every horizon, from the most recent training run
- `.github/workflows/hourly_features.yml` — automated hourly data collection
- `.github/workflows/retrain_model.yml` — automated model retraining every 4 days

## Known limitations

- OpenWeather's underlying atmospheric model shows large, city-specific bias against real ground-truth stations (up to ~2.3x under/over-estimation); this is corrected only for the 4 cities with AQICN validation data. The other 26 cities have no ground-truth station and remain uncorrected — documented as a data source limitation rather than guessed at.
- The neural network compared during model selection was trained once on Colab and is not part of the automated retraining pipeline; only the 3 production models (RandomForest, Ridge, XGBoost) retrain automatically.

## Stack

Python · scikit-learn · XGBoost · TensorFlow/Keras (comparison only) · Hopsworks · GitHub Actions · Streamlit · Plotly · SHAP · joblib
