# Intelligent Telecom Operations & Optimization Platform

This project builds an end-to-end big data ML system tailored for the telecom industry. It integrates:

- 📈 **Time Series Forecasting** for traffic/load per cell tower
- 🛑 **Anomaly Detection** to flag unusual activity or failures
- 🔁 **Recommender Systems** for churn prevention and plan upgrades
- 🔗 **Graph ML** for analyzing the telecom network graph
- 🧠 **Variational Autoencoders (VAE)** for generative modeling & anomaly scoring
- ⚡ **PySpark** to handle large-scale data processing of CDRs and network logs

## 🚀 Features
- Forecast regional traffic using LSTM/TFT
- Detect anomalies using Autoencoders and Isolation Forests
- Train GNNs to model device-user networks
- Generate recommendations using Spark ALS or hybrid models
- Build a live dashboard with Streamlit

## 🧰 Tech Stack
- Python, PySpark, Pandas, Scikit-learn
- PyTorch, PyTorch Lightning, PyTorch Geometric
- MLlib, Streamlit, NetworkX, Neo4j
- Jupyter, YAML for config

## 📁 Folder Structure

telecom-intelligent-platform/
│
├── data/
│   ├── raw/                   # Original or simulated CDRs, traffic logs, etc.
│   ├── processed/             # Cleaned, transformed datasets
│   └── external/              # Any downloaded public datasets
│
├── notebooks/                 # Jupyter notebooks for exploration & prototyping
│   ├── eda/
│   ├── forecasting/
│   ├── anomaly_detection/
│   └── graph_modeling/
│
├── src/
│   ├── data_ingestion/        # PySpark ETL jobs
│   │   ├── load_cdrs.py
│   │   ├── transform_logs.py
│   │   └── utils.py
│   │
│   ├── forecasting/           # Time series forecasting models
│   │   ├── models.py
│   │   └── train.py
│   │
│   ├── anomaly_detection/     # VAE, Isolation Forests, etc.
│   │   ├── train_vae.py
│   │   ├── detect_anomalies.py
│   │   └── evaluate.py
│   │
│   ├── recommender/           # PySpark ALS, hybrid recommenders
│   │   ├── train_als.py
│   │   ├── evaluate.py
│   │   └── utils.py
│   │
│   ├── graph_ml/              # Graph construction & GNN models
│   │   ├── build_graph.py
│   │   ├── gnn_model.py
│   │   └── train_gnn.py
│   │
│   └── vae_module/            # Shared VAE models for anomaly & gen
│       ├── vae.py
│       └── train.py
│
├── dashboard/                 # Streamlit or Dash app
│   ├── app.py
│   └── components/
│
├── config/                    # Config files (YAML/JSON) for pipeline params
│   ├── spark_config.yaml
│   ├── model_config.yaml
│   └── dashboard_config.yaml
│
├── scripts/                   # Bash or Python scripts for running jobs
│   ├── run_pipeline.sh
│   └── launch_dashboard.sh
│
├── tests/                     # Unit and integration tests
│   ├── test_forecasting.py
│   ├── test_recommender.py
│   └── ...
│
├── requirements.txt           # Python + PySpark dependencies
├── environment.yml            # Conda environment (if using)
├── README.md
└── LICENSE

## 📊 **Project Data Plan**

| **Module**                | **Dataset Needed**                                                     | **Source**                          | **Purpose**                                                      |
|---------------------------|------------------------------------------------------------------------|-------------------------------------|------------------------------------------------------------------|
| **1. Time Series Forecasting** | Telecom traffic volume by region/tower/time                             | ✅ Synthetic (generated) or 📦 OpenCelliD + simulated usage | Predict traffic spikes, enable pre-scaling of resources         |
| **2. Anomaly Detection**      | Sensor logs / usage logs with anomalies                                  | ✅ Synthetic with injected anomalies | Detect performance drops, network failures                      |
| **3. Graph ML**               | Device ↔ User ↔ Tower relationships (telecom graph)                      | ✅ Synthetic or 📦 MIT Reality Mining / D4D | Model failure propagation, optimize network topology            |
| **4. Recommender System**     | Customer usage history, plan info, churn flags                           | 📦 IBM Telecom Churn Dataset or ✅ Simulated | Recommend upgrades, predict churn                               |
| **5. VAE Module**             | Normal network behavior logs or encoded traffic sequences                | Derived from anomaly dataset        | Learn compressed latent space, generate synthetic behavior data |
| **6. PySpark Ingestion**      | Raw CDRs (Call Detail Records), usage logs, support ticket logs          | ✅ Simulated or 📦 D4D / Alibaba traces | Handle large-scale ingestion, cleaning, transformations         |
| **7. Dashboard**              | Combined data outputs (forecasts, anomalies, graphs, recs)               | Internal project outputs            | Visualization & interaction with the system                     |

## Usage

1. Clone the repository
2. Generate the data using the script scripts/generate_data.py
3. Place the [IBM Telco Churn](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) Kaggle dataset under data/external
4. Run the script scripts/process_all_data.py using the Docker container.
docker compose build
docker compose up
docker-compose exec pyspark-notebook python work/scripts/process_all_data.py

docker exec -it telecom-intelligent-ml-platform-pyspark-notebook-1 bash