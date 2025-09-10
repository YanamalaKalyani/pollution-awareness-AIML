import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(path="data/environment_data.csv"):
    data = pd.read_csv(path)

    # Fill missing values
    data.fillna(method="ffill", inplace=True)

    # Encode categorical target if necessary
    if data["target"].dtype == "object":
        le = LabelEncoder()
        data["target"] = le.fit_transform(data["target"])

    # Features & Target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess()
    print("Preprocessing complete. X shape:", X.shape)