# 📊 Telecom Analytics Dashboard (Streamlit)
# Save this as: app/dashboard_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

from churn_model_utils import FEATURES, load_model as load_churn_model, predict as predict_churn
from anomaly_model_utils import load_model as load_anomaly_model, predict_anomalies
from forecasting_model_utils import load_model as load_forecast_model, forecast as make_forecast, load_data as load_forecast_data
from anomaly_model_utils import FEATURES as ANOMALY_FEATURES

st.set_page_config(page_title="Telecom Dashboard", layout="wide")
sns.set(style="whitegrid")
PROCESSED_PATH = "./data/processed/parquet/"

# Load Data
@st.cache_data
def load_data():
    user_df = pd.read_parquet(PROCESSED_PATH+"user_feature_matrix.parquet")
    tower_df = pd.read_parquet(PROCESSED_PATH+"tower_feature_matrix.parquet")
    traffic_df = pd.read_parquet(PROCESSED_PATH+"traffic_volume.parquet")
    anomaly_df = pd.read_parquet(PROCESSED_PATH+"network_behaviors.parquet")
    return user_df, tower_df, traffic_df, anomaly_df

user_df, tower_df, traffic_df, anomaly_df = load_data()

# Load Churn Model
@st.cache_resource
def load_churn_assets():
    model, scaler = load_churn_model("./models/churn_model.pkl", "./models/churn_scaler.pkl")
    return model, scaler

model_churn, scaler_churn = load_churn_assets()

# Load Anomaly Model
@st.cache_resource
def load_anomaly_assets():
    model, scaler = load_anomaly_model("./models/anomaly_model.pkl", "./models/anomaly_scaler.pkl")
    return model, scaler

model_anomaly, scaler_anomaly = load_anomaly_assets()

# Load Forecasting Model
@st.cache_resource
def load_forecast_assets():
    model = load_forecast_model("./models/forecast_model.pkl")
    return model

model_forecast = load_forecast_assets()

# Sidebar Navigation
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["User Overview", "Network Overview", "Traffic Trends", "Anomalies", "Churn Prediction", "Forecasting"])

# User Overview Page
if page == "User Overview":
    st.title("👥 User Insights")
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="churned", data=user_df, ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

    st.write("### Usage vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="churned", y="avg_data", data=user_df, ax=ax)
    st.pyplot(fig)

    with st.expander("📄 View Raw User Data"):
        st.dataframe(user_df.head())

# Network Overview Page
elif page == "Network Overview":
    st.title("📡 Tower Load Overview")
    fig, ax = plt.subplots()
    sc = ax.scatter(
        tower_df["longitude"],
        tower_df["latitude"],
        c=tower_df["avg_calls"],
        cmap="viridis",
        s=60,
        alpha=0.8
    )
    plt.colorbar(sc, ax=ax, label="Avg Calls")
    ax.set_title("Tower Locations Colored by Average Call Volume")
    st.pyplot(fig)

    with st.expander("📄 View Raw Tower Data"):
        st.dataframe(tower_df.head())

# Traffic Trends Page
elif page == "Traffic Trends":
    st.title("📈 Traffic Trends by Hour")
    traffic_df["hour"] = pd.to_datetime(traffic_df["hour"])
    traffic_df["hour_of_day"] = traffic_df["hour"].dt.hour
    hourly_avg = traffic_df.groupby("hour_of_day")["total_calls"].mean()

    fig, ax = plt.subplots()
    hourly_avg.plot(kind="line", ax=ax)
    ax.set_title("Average Calls by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Avg Calls")
    st.pyplot(fig)

    with st.expander("📄 View Raw Traffic Data"):
        st.dataframe(traffic_df.head())

# Anomalies
if page == "Anomalies":
    st.title("⚠️ Anomaly Detection")
    X = anomaly_df[ANOMALY_FEATURES]
    X_scaled = scaler_anomaly.transform(X)
    preds = predict_anomalies(model_anomaly, X_scaled)
    anomaly_df["predicted_anomaly"] = preds

    fig, ax = plt.subplots()
    sns.kdeplot(data=anomaly_df[anomaly_df.predicted_anomaly == 0]["latency_ms"], label="Normal", ax=ax)
    sns.kdeplot(data=anomaly_df[anomaly_df.predicted_anomaly == 1]["latency_ms"], label="Predicted Anomaly", ax=ax)
    ax.set_title("Latency Distribution by Predicted Anomaly")
    ax.set_xlabel("Latency (ms)")
    ax.legend()
    st.pyplot(fig)

    with st.expander("📄 View Predicted Anomalies"):
        st.dataframe(anomaly_df[anomaly_df.predicted_anomaly == 1])

# Churn Prediction Page
if page == "Churn Prediction":
    st.title("📉 Churn Prediction")
    st.write("Select users to view predicted churn risk.")

    # Filters
    selected_region = st.selectbox("Filter by Region", options=["All"] + sorted(user_df["region_id"].dropna().unique().tolist()))

    filtered_df = user_df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region_id"] == selected_region]

    if not filtered_df.empty:
        probs = predict_churn(model_churn, scaler_churn, filtered_df)
        filtered_df["churn_probability"] = probs

        st.write("### Users sorted by predicted churn probability")
        st.dataframe(filtered_df[["user_id", "region_id", "avg_data", "avg_minutes", "avg_satisfaction", "churn_probability"]]
                     .sort_values(by="churn_probability", ascending=False).reset_index(drop=True))
    else:
        st.warning("No users available for selected filter.")

# Forecasting Page
if page == "Forecasting":
    st.title("📈 Traffic Forecast")

    forecast_horizon = st.slider("Forecast Hours Ahead", min_value=12, max_value=72, step=12, value=24)
    traffic_data = load_forecast_data(PROCESSED_PATH+"traffic_volume.parquet")
    forecast_df = make_forecast(model_forecast, periods=forecast_horizon)

    st.write("### Forecast vs. Actual")
    actual = traffic_data.set_index("ds")["y"]
    predicted = forecast_df.set_index("ds")["yhat"]
    common_index = actual.index.intersection(predicted.index)

    plt.figure(figsize=(10, 4))
    plt.plot(actual.loc[common_index], label="Actual")
    plt.plot(predicted.loc[common_index], label="Forecast")
    plt.title("Traffic Forecast vs. Actual")
    plt.xlabel("Time")
    plt.ylabel("Call Volume")
    plt.legend()
    st.pyplot(plt.gcf())
