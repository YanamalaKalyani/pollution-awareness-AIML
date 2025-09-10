# src/train_model.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess  # same folder import

def train_and_save(
    path_data="data/environment_data.csv",
    model_out="models/trained_model.pkl",
    scaler_out="models/scaler.pkl"
):
    # 1️⃣ Load and preprocess data
    X, y, scaler = load_and_preprocess(path_data)

    # 2️⃣ Convert target to string (fixes mixed type issues)
    y = y.astype(str)

    # 3️⃣ Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5️⃣ Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).astype(str)  # ensure consistent type

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6️⃣ Save model and scaler
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)
    print(f"Saved model -> {model_out}")
    print(f"Saved scaler -> {scaler_out}")

if __name__ == "__main__":
    train_and_save()
