import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─────────────────────────────
# Setup Spark session
# ─────────────────────────────
spark = SparkSession.builder.appName("TelecomDataProcessing").getOrCreate()

RAW_PATH = "work/data/raw/"
PROCESSED_PATH = "work/data/processed/"

def save_dataframe(df, path, format="parquet", single_file=True, overwrite=True):
    """
    Save Spark DataFrame to disk in an easy-to-process format.

    Args:
        df (DataFrame): Spark DataFrame to save.
        path (str): Destination path (e.g., 'data/processed/users').
        format (str): 'csv' or 'parquet'.
        single_file (bool): If True, output will be a single file (coalesce + move).
        overwrite (bool): If True, overwrite existing files.

    Returns:
        str: Final path to the saved file or directory.
    """

    if overwrite and os.path.exists(path):
        shutil.rmtree(path)

    # Coalesce to 1 partition if single file desired
    writer = df.coalesce(1) if single_file else df

    # Set writer options
    writer = writer.write.option("header", True)
    if overwrite:
        writer = writer.mode("overwrite")

    # Write based on format
    if format == "csv":
        writer.csv(path)
    elif format == "parquet":
        writer.parquet(path)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")

    # If CSV and single_file=True, rename part file
    if format == "csv" and single_file:
        part_file = None
        for fname in os.listdir(path):
            if fname.startswith("part-") and fname.endswith(".csv"):
                part_file = fname
                break
        if part_file:
            final_path = path + ".csv"
            shutil.move(os.path.join(path, part_file), final_path)
            shutil.rmtree(path)
            return final_path

    return path

# ─────────────────────────────
# 1. Traffic Volume (per tower/hour)
# ─────────────────────────────
cdr = spark.read.csv(RAW_PATH + "synthetic_cdr.csv", header=True, inferSchema=True)

traffic = cdr.withColumn("hour", F.date_trunc("hour", F.col("timestamp"))) \
    .groupBy("tower_id", "hour") \
    .agg(
        F.count("*").alias("total_calls"),
        F.sum("data_usage_mb").alias("total_data_used_mb")
    )

save_dataframe(traffic, PROCESSED_PATH + "traffic_volume", format="csv")

# ─────────────────────────────
# 2. User-Tower Graph Edges
# ─────────────────────────────
edges = cdr.groupBy("user_id", "tower_id") \
    .agg(F.count("*").alias("interaction_weight"))

save_dataframe(edges, PROCESSED_PATH + "telecom_graph_edges", format="csv")

# ─────────────────────────────
# 3. Tower Anomaly Metrics
# ─────────────────────────────
logs = spark.read.csv(RAW_PATH + "anomaly_logs.csv", header=True, inferSchema=True)

network_stats = logs.select("latency_ms", "packet_loss", "cpu_load", "anomaly")

save_dataframe(network_stats, PROCESSED_PATH + "network_behaviors", format="csv")

# ─────────────────────────────
# 4. User Churn Aggregated Features
# ─────────────────────────────
plans = spark.read.csv(PROCESSED_PATH + "user_plan_history.csv", header=True, inferSchema=True)

churn_ds = plans.groupBy("user_id") \
    .agg(
        F.max("churn_flag").alias("churned"),
        F.avg("minutes_used").alias("avg_minutes"),
        F.avg("data_used_gb").alias("avg_data_gb"),
        F.avg("satisfaction_score").alias("avg_satisfaction")
    )

save_dataframe(churn_ds, PROCESSED_PATH + "user_churn_dataset", format="csv")

# ─────────────────────────────
# 5. Copy Raw Metadata Files (User & Tower)
# ─────────────────────────────
files_to_copy = [
    (RAW_PATH + "user_profiles.csv", PROCESSED_PATH + "user_profiles.csv"),
    (RAW_PATH + "tower_locations.csv", PROCESSED_PATH + "tower_locations.csv")
]

os.makedirs(PROCESSED_PATH, exist_ok=True)

for src, dst in files_to_copy:
    shutil.copyfile(src, dst)
    print(f"✅ Copied: {src} → {dst}")

print("✅ All processed data generated.")
