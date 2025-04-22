import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = ["avg_minutes", "avg_data", "avg_satisfaction", "age", "is_enterprise"]
TARGET = "churned"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_model(X, y):
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.2f}")


def save_model(model, scaler, model_path: str, scaler_path: str):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, input_df: pd.DataFrame):
    input_scaled = scaler.transform(input_df[FEATURES])
    return model.predict_proba(input_scaled)[:, 1]
