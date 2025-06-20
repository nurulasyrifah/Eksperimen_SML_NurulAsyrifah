import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import shutil

mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Tracking ke MLflow lokal
mlflow.set_experiment("Valve Failure Prediction")

# Load data
df = pd.read_csv("valveplatefailure_preprocessing/valve_plate_clean_automate.csv")

# Preprocessing sederhana
X = df.drop(columns=["label", "stan", "Czas", "Czas2"], errors='ignore')
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Aktifkan autolog
mlflow.sklearn.autolog()

# Training
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    model_path = "rf_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Simpan model eksplisit ke folder lokal juga
    mlflow.sklearn.save_model(model, "rf_model")
