# 📡 Telecom ML Intelligence Platform

An end-to-end machine learning platform for telecom analytics featuring:
- 📉 **Churn Prediction**
- ⚠️ **Anomaly Detection**
- 🔮 **Traffic Forecasting**
- 📊 **Interactive Streamlit Dashboard** for visualization and insights

---

## 🧠 Project Highlights

- **Synthetic telecom dataset generation** including CDRs, user profiles, network behavior logs, and plan history
- **PySpark preprocessing** simulated within a Dockerized Jupyter environment
- **EDA** using Pandas, Matplotlib, and Seaborn
- **Model training** for:
  - Churn prediction using Random Forest + SMOTE
  - Network anomaly detection via Isolation Forest
  - Hourly traffic forecasting using Prophet
- **Modular code** in `src/` for clean ML pipelines, utilities, and inference
- **Streamlit dashboard** to interactively explore users, towers, traffic trends, and ML predictions

---

## 🗂️ Directory Structure

```
├── data/
│   ├── raw/                     # Generated raw synthetic datasets
│   └── processed/               # Cleaned Parquet datasets
├── models/
│   └── *.pkl                    # Trained model files
├── notebooks/
│   └── 01_process_raw_to_processed.ipynb
│   └── 02_create_features.ipynb
│   └── 03_eda_analysis.ipynb
│   └── 04_model_churn_prediction.ipynb
│   └── 05_model_anomaly_detection.ipynb
│   └── 06_model_forecasting.ipynb
├── scripts/
│   ├── generate_data.py         # Data generation logic
│   └── process_all_data.py      # Data processing logic
└── src/
    ├── dashboard_app.py         # Streamlit dashboard
    ├── churn_model_utils.py         
    ├── anomaly_model_utils.py       
    └── forecasting_model_utils.py   
```

---

## 🗃️ Data Sources

All datasets used in this project are **synthetically generated** to simulate a realistic telecom environment:

- `synthetic_cdr.csv`: Call Detail Records with timestamps, tower usage, data and call volume
- `user_profiles.csv`: User demographics like age, region, and enterprise flags
- `user_plan_history.csv`: Monthly data usage and satisfaction scores
- `network_behaviors.csv`: Tower-level performance logs (latency, CPU, packet loss)
- `tower_locations.csv`: Geolocation and region info for each tower
- `ibm_telco_churn.csv`: Kaggle Dataset [IBM Telco Churn](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

These datasets are joined, aggregated, and saved as Parquet using PySpark within a **Docker-based environment** to simulate distributed processing.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python scripts/generate_data.py
```

### 3. Process Data (Docker is optional if pyspark not supported)
```bash
docker compose build
docker compose up
docker-compose exec pyspark-notebook python work/scripts/process_all_data.py
```

### 4. Launch Dashboard
```bash
streamlit run src/dashboard_app.py
```

---

## 📌 Use Cases Covered

- Identify high-risk users based on churn probabilities
- Detect underperforming or anomalous towers
- Forecast future call volume to support capacity planning

---

## 🏁 Next Steps

- [ ] Add CSV upload support for new user churn predictions
- [ ] Add map visualizations for tower anomalies
- [ ] Add model monitoring features (e.g., prediction drift detection)

---

**Author:** Daniel James