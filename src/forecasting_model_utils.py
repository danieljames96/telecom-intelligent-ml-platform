# src/models/forecasting_model_utils.py

import pandas as pd
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df_agg = df.groupby("hour")["total_calls"].sum().reset_index()
    df_agg.columns = ["ds", "y"]
    df_agg["ds"] = pd.to_datetime(df_agg["ds"])
    return df_agg


def train_model(df_train: pd.DataFrame) -> Prophet:
    model = Prophet()
    model.fit(df_train)
    return model


def forecast(model: Prophet, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq="H")
    forecast_df = model.predict(future)
    return forecast_df


def evaluate(df_true: pd.DataFrame, forecast_df: pd.DataFrame):
    pred = forecast_df.set_index("ds").loc[df_true["ds"]]["yhat"]
    true = df_true.set_index("ds")["y"]
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")


def save_model(model: Prophet, path: str):
    joblib.dump(model, path)


def load_model(path: str) -> Prophet:
    return joblib.load(path)