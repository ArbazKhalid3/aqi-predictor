# Pakistan AQI Predictor

A serverless, end-to-end air quality forecasting system built for the 10Pearls internship project brief. Predicts AQI up to 3 days ahead for 30 Pakistani cities using an automated hourly data pipeline, a Hopsworks feature store, and a trained ML model served through a live dashboard.

**Live dashboard:** https://aqi-predictor-ydr7bnqs9d5twyutmwjhep.streamlit.app

## What it does

- Collects hourly weather + air pollutant data for 30 Pakistani cities from OpenWeather's Air Pollution API
- Computes standardized US EPA AQI from raw pollutant concentrations (PM2.5, PM10)
- Cross-validates against real AQICN ground-station readings for 4 cities (Karachi, Lahore, Islamabad, Peshawar), and applies a per-city bias correction where validation data supports it
- Stores features in a Hopsworks feature store (in addition to a local CSV)
- Trains a RandomForest model to predict AQI 72 hours ahead, compared against a Ridge baseline and a Colab-trained feedforward neural network
- Explains model predictions with SHAP feature importance
- Displays live readings, historical trends, and 3-day forecasts on an interactive map-based dashboard

## Architecture

OpenWeather API + AQICN API feed an hourly feature pipeline (automated via GitHub Actions), which writes to both a Hopsworks feature store and a local CSV. That data feeds a RandomForest training pipeline, which powers the live Streamlit dashboard's forecast.

## Project structure

- `scripts/` — data fetching, feature engineering, model training, SHAP explainability, calibration analysis
- `app/dashboard.py` — the Streamlit dashboard
- `notebooks/` — EDA and calibration analysis notebooks (run as `# %%` cells in VS Code)
- `data/` — feature store CSV, trained models
- `.github/workflows/` — hourly automated data collection pipeline

## Known limitations

- OpenWeather's underlying atmospheric model (SILAM) shows large, city-specific bias against real ground-truth stations; this is corrected only for the 4 cities with AQICN validation data. The other 26 cities have no ground-truth station and remain uncorrected — documented as a data source limitation rather than guessed at.
- Some EDA findings (day-of-week pattern, hourly pattern) were noted as preliminary when collected with under a week of data.

## Stack

Python · scikit-learn · TensorFlow/Keras · Hopsworks · GitHub Actions · Streamlit · Plotly · SHAP
