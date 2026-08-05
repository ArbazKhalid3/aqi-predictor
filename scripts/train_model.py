"""
Training pipeline:
1. Loads historical (features, targets) from our feature store (features.csv for now).
2. Builds the actual prediction target: AQI N hours in the future, per city,
   using real elapsed time (not just "next row") so it stays correct even
   if the hourly schedule has gaps or irregular timestamps.
3. Trains + evaluates a few models, picks the best by RMSE.
4. Saves the trained model to the model registry (local file for now --
   we'll swap this for Hopsworks once the pipeline logic is proven).
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = "../data/features.csv"
MODEL_OUT = "../data/model.joblib"

# Brief asks for a 3-day-ahead forecast. Kept as a constant so it's easy
# to shrink temporarily (e.g. to 1 hour) just to test the pipeline logic
# early, before we have 3 real days of history.
FORECAST_HORIZON_HOURS = 72

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def build_training_set(df):
    """
    For each city, match every row to the reading closest to
    (its timestamp + FORECAST_HORIZON_HOURS) and use that future AQI as
    the label. Rows with no valid future match (not enough history yet)
    are dropped -- this is why early runs will have few/zero rows.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    training_rows = []

    for city, group in df.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for i, row in group.iterrows():
            target_time = row["timestamp"] + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
            # allow a +/- 30 min tolerance window around the target time
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
    X = train_df[FEATURE_COLUMNS].fillna(0)  # missing pollutants (e.g. Karachi's O3) -> 0 for now
    y = train_df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Ridge": Ridge(alpha=1.0),
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
