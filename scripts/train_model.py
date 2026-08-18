"""
Training pipeline -- now trains THREE models, one per forecast horizon
(24h/48h/72h), per the brief's request to predict "the next 3 days"
(interpreted as a day-by-day forecast, not a single 72h-ahead number).

1. Loads historical (features, targets) from our feature store.
2. For each horizon, builds (features, target) pairs and trains
   RandomForest, Ridge, and XGBoost, picking the best by RMSE.
3. Saves all three horizon models together in one file, keyed by horizon.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = "../data/features.csv"
MODEL_OUT = "../data/model.joblib"

FORECAST_HORIZONS_HOURS = [24, 48, 72]  # Day 1, Day 2, Day 3

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def build_training_set(df, horizon_hours):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    training_rows = []

    for city, group in df.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i, row in group.iterrows():
            target_time = row["timestamp"] + pd.Timedelta(hours=horizon_hours)
            window = group[
                (group["timestamp"] >= target_time - pd.Timedelta(minutes=30)) &
                (group["timestamp"] <= target_time + pd.Timedelta(minutes=30))
            ]
            if window.empty:
                continue
            feature_row = row[FEATURE_COLUMNS].to_dict()
            feature_row["target_aqi"] = window.iloc[0]["aqi"]
            training_rows.append(feature_row)

    return pd.DataFrame(training_rows)


def train_and_evaluate(train_df, horizon_hours):
    X = train_df[FEATURE_COLUMNS].fillna(0)
    y = train_df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=3, random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}
        print(f"  {name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"  Best for {horizon_hours}h: {best_name}")
    return results[best_name]["model"], best_name


def run():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} raw rows from feature store.")

    all_models = {}
    for horizon in FORECAST_HORIZONS_HOURS:
        print(f"\n=== Training {horizon}h-ahead model ===")
        train_df = build_training_set(df, horizon)
        print(f"Built {len(train_df)} training rows for {horizon}h horizon.")

        if len(train_df) < 20:
            print(f"Not enough data yet for {horizon}h horizon, skipping.")
            continue

        best_model, best_name = train_and_evaluate(train_df, horizon)
        all_models[horizon] = {"model": best_model, "name": best_name, "features": FEATURE_COLUMNS}

    if not all_models:
        print("\nNo horizon had enough data to train. Let the pipeline keep running.")
        return

    joblib.dump(all_models, MODEL_OUT)
    print(f"\nSaved {len(all_models)} horizon model(s) to {MODEL_OUT}")
    for h, info in all_models.items():
        print(f"  {h}h -> {info['name']}")


if __name__ == "__main__":
    run()
