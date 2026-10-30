"""Yield model utilities.

A genuine yield model requires historical labelled yield data. The bundled sample
is synthetic and is used only to demonstrate the end-to-end ML workflow.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

MODEL_PATH = Path("data/yield_model.joblib")
SAMPLE_PATH = Path("data/sample_yield_training.csv")

FEATURES = [
    "seasonal_rainfall_mm", "mean_temp_c", "soil_moisture",
    "ndvi", "crop_encoded"
]
TARGET = "yield_tonnes_per_hectare"

def create_demo_dataset(path=SAMPLE_PATH, rows=700, seed=17):
    rng = np.random.default_rng(seed)
    rain = rng.uniform(150, 1200, rows)
    temp = rng.uniform(18, 40, rows)
    moisture = rng.uniform(0.08, 0.48, rows)
    ndvi = rng.uniform(0.15, 0.9, rows)
    crop = rng.integers(0, 4, rows)
    # Synthetic relationship purely for demonstration, not agronomic truth.
    yield_value = (
        1.2 + 0.0035 * np.minimum(rain, 750)
        - 0.075 * np.maximum(temp - 30, 0)
        + 5.0 * moisture + 3.2 * ndvi
        + crop * 0.35 + rng.normal(0, 0.45, rows)
    )
    yield_value = np.clip(yield_value, 0.2, None)
    df = pd.DataFrame({
        "seasonal_rainfall_mm": rain,
        "mean_temp_c": temp,
        "soil_moisture": moisture,
        "ndvi": ndvi,
        "crop_encoded": crop,
        TARGET: yield_value
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def train_model(csv_path=None):
    path = Path(csv_path) if csv_path else SAMPLE_PATH
    if not path.exists():
        create_demo_dataset(path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Training CSV is missing columns: " + ", ".join(missing))
    df = df[FEATURES + [TARGET]].dropna()
    if len(df) < 30:
        raise ValueError("Please provide at least 30 complete labelled rows.")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rows": int(len(df)),
        "model": model
    }

def predict_yield(features: dict, csv_path=None):
    path = Path(csv_path) if csv_path else SAMPLE_PATH
    if not MODEL_PATH.exists():
        train_model(path)
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{k: features[k] for k in FEATURES}], columns=FEATURES)
    return float(model.predict(row)[0])
