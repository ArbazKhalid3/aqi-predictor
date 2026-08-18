"""
Training pipeline:
1. Loads historical (features, targets) from our feature store (features.csv for now).
2. Builds the actual prediction target: AQI N hours in the future, per city,
   using real elapsed time (not just "next row") so it stays correct even
   if the hourly schedule has gaps or irregular timestamps.
3. Trains + evaluates several models (RandomForest, Ridge, XGBoost), picks
   the best by RMSE -- per the brief's request for "a variety of forecasting
   models, from statistical modelling to deep learning."
4. Saves the trained model to the model registry (local file for now --
   we'll swap this for Hopsworks once the pipeline logic is proven).
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

FORECAST_HORIZON_HOURS = 72

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def build_training_set(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    training_rows = []

    for city, group in df.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i, row in group.iterrows():
            target_time = row["timestamp"] + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
            window = group[
                (group["timestamp"] >= target_time - pd.Timedelta(minutes=30)) &
                (group["timestamp"] <= target_time + pd.Timedelta(minutes=30))
            ]
            if window.empty:
                continue
            future_aqi = window.iloc[0]["aqi"]

            feature_row = row[FEATURE_COLUMNS].to_dict()
            feature_row["target_aqi"] = future_aqi
            training_rows.append(feature_row)

    return pd.DataFrame(training_rows)


def train_and_evaluate(train_df):
    X = train_df[FEATURE_COLUMNS].fillna(0)
    y = train_df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
       "RandomForest": RandomForestRegressor(
            n_estimators=100,       # fewer trees -- halves model file size, minimal accuracy cost
            max_depth=15,           # caps tree depth -- keeps file size bounded as data keeps growing
            min_samples_leaf=3,     # slightly smoother trees, also reduces overfitting risk
            random_state=42,
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
        print(f"{name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"\nBest model: {best_name}")
    return results[best_name]["model"], best_name


def run():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} raw rows from feature store.")

    train_df = build_training_set(df)
    print(f"Built {len(train_df)} (features, target) training rows "
          f"[needs pairs {FORECAST_HORIZON_HOURS}h apart per city].")

    if len(train_df) < 20:
        print(
            f"\nNot enough historical data yet to train a real model. "
            f"Need more hourly runs to accumulate before pairs {FORECAST_HORIZON_HOURS}h "
            f"apart exist. Let the GitHub Actions pipeline keep running and re-run this "
            f"script again in a few days."
        )
        return

    best_model, best_name = train_and_evaluate(train_df)
    joblib.dump({"model": best_model, "name": best_name, "features": FEATURE_COLUMNS}, MODEL_OUT)
    print(f"Saved best model ({best_name}) to {MODEL_OUT}")


if __name__ == "__main__":
    run()
