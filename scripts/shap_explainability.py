"""
SHAP explainability: shows which features actually drive the RandomForest
model's 3-day-ahead AQI predictions, and by how much. Required by the
project brief ("use SHAP or LIME for feature importance explanations").
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

MODEL_PATH = "../data/model.joblib"
FEATURES_CSV = "../data/features.csv"
FORECAST_HORIZON_HOURS = 72

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temp_c", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def build_training_set(df):
    """Same logic as train_model.py -- rebuilds the (features, target) pairs
    so SHAP has the exact same data the model was trained on."""
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
            feature_row = row[FEATURE_COLUMNS].to_dict()
            feature_row["target_aqi"] = window.iloc[0]["aqi"]
            training_rows.append(feature_row)
    return pd.DataFrame(training_rows)


print("Loading model and rebuilding training data...")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
print(f"Loaded model: {saved['name']}")

df = pd.read_csv(FEATURES_CSV)
train_df = build_training_set(df)
X = train_df[FEATURE_COLUMNS].fillna(0)
print(f"Explaining predictions on {len(X)} rows")

# TreeExplainer is fast and exact for tree-based models like RandomForest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global feature importance: which features matter most, on average,
# across all predictions
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Feature Importance (mean |SHAP value|)")
plt.tight_layout()
plt.savefig("../notebooks/plots_shap_importance.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: plots_shap_importance.png")

# Beeswarm plot: shows not just importance, but the DIRECTION of each
# feature's effect (e.g. does higher PM2.5 push AQI prediction up or down)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary — Feature Impact on Predicted AQI")
plt.tight_layout()
plt.savefig("../notebooks/plots_shap_summary.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved: plots_shap_summary.png")

# Print numeric ranking too, useful for the report text
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_ranking = pd.Series(mean_abs_shap, index=FEATURE_COLUMNS).sort_values(ascending=False)
print("\nFeature importance ranking:")
print(importance_ranking)
