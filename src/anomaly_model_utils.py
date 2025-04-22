import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ["latency_ms", "packet_loss", "cpu_load"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame):
    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_model(X_scaled):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_scaled)
    return model


def predict_anomalies(model, X_scaled):
    preds = model.predict(X_scaled)
    return (preds == -1).astype(int)  # 1 = anomaly, 0 = normal


def evaluate_model(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def save_model(model, scaler, model_path: str, scaler_path: str):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler