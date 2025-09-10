# src/evaluate.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(
    path_data="data/environment_data.csv",
    model_in="models/trained_model.pkl",
    scaler_in="models/scaler.pkl"
):
    # 1️⃣ Load dataset
    df = pd.read_csv(path_data)
    label_column = "target"  # replace with your target column name
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    # 2️⃣ Convert target to string (fix mixed type issue)
    y = y.astype(str)

    # 3️⃣ Split into test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4️⃣ Load saved model and scaler
    model = joblib.load(model_in)
    scaler = joblib.load(scaler_in)

    # 5️⃣ Scale test features
    X_test_scaled = scaler.transform(X_test)

    # 6️⃣ Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred = pd.Series(y_pred).astype(str)  # ensure consistent type

    # 7️⃣ Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 8️⃣ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # 9️⃣ Confusion Matrix Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    # 🔟 Feature Importance (Random Forest only)
    if hasattr(model, "feature_importances_"):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8,6))
        sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()

if __name__ == "__main__":
    evaluate()
