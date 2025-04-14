import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from faker import Faker

def generate_synthetic_cdr(num_users=500, num_towers=50, num_regions=10, num_days=7, records_per_user_per_day=10, output_path="data/raw/synthetic_cdr.csv"):
    start_date = datetime(2025, 3, 25)
    all_records = []

    for user_id in range(1, num_users + 1):
        region_id = random.randint(1, num_regions)
        for day in range(num_days):
            for _ in range(records_per_user_per_day):
                timestamp = start_date + timedelta(days=day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
                tower_id = random.randint(1, num_towers)
                call_duration = round(np.random.exponential(scale=3), 2)  # minutes
                data_usage_mb = round(np.random.exponential(scale=50), 2)  # MB
                dropped_call = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% chance
                roaming = np.random.choice([0, 1], p=[0.9, 0.1])
                
                all_records.append([
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    f"U{user_id:04d}",
                    f"T{tower_id:03d}",
                    call_duration,
                    data_usage_mb,
                    dropped_call,
                    region_id,
                    roaming
                ])

    df = pd.DataFrame(all_records, columns=[
        "timestamp", "user_id", "tower_id", "call_duration",
        "data_usage_mb", "dropped_call", "region_id", "roaming"
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic CDR dataset saved to: {output_path}")
    print(df.head())
    
def generate_user_plan_history(num_users=500, months=6, output="data/processed/user_plan_history.csv"):
    rows = []
    start = datetime(2024, 10, 1)

    for user_id in range(1, num_users + 1):
        churn_flag = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% churn rate
        plan_type = np.random.choice(['Basic', 'Standard', 'Premium'], p=[0.4, 0.4, 0.2])

        for m in range(months):
            ts = start + pd.DateOffset(months=m)
            mins_used = np.random.poisson(300)
            data_used = np.random.exponential(scale=8)
            satisfaction = round(np.clip(np.random.normal(3.5, 0.8), 1, 5), 1)

            rows.append([f"U{user_id:04d}", ts.strftime("%Y-%m"), plan_type,
                         mins_used, round(data_used, 2), satisfaction, churn_flag])

    df = pd.DataFrame(rows, columns=[
        "user_id", "month", "plan_type", "minutes_used", "data_used_gb", "satisfaction_score", "churn_flag"
    ])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✅ Saved: {output}")

def generate_anomaly_logs(num_towers=50, days=7, output="data/raw/anomaly_logs.csv"):
    start_time = datetime(2025, 3, 25)
    rows = []

    for tower_id in range(1, num_towers + 1):
        for h in range(24 * days):
            ts = start_time + timedelta(hours=h)
            latency = np.random.normal(100, 15)
            packet_loss = np.random.normal(0.01, 0.005)
            cpu_load = np.random.normal(0.5, 0.1)

            # Inject random anomalies
            if random.random() < 0.02:
                latency *= 2
                packet_loss += 0.1
                cpu_load += 0.4
                anomaly = 1
            else:
                anomaly = 0

            rows.append([ts, f"T{tower_id:03d}", latency, packet_loss, cpu_load, anomaly])

    df = pd.DataFrame(rows, columns=["timestamp", "tower_id", "latency_ms", "packet_loss", "cpu_load", "anomaly"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✅ Saved: {output}")
    
def generate_tower_locations(num_towers=50, regions=10, output="data/raw/tower_locations.csv"):
    towers = []
    for i in range(1, num_towers + 1):
        lat = np.random.uniform(5.0, 15.0)    # Fake telecom region (Africa/Asia)
        lon = np.random.uniform(70.0, 85.0)
        region_id = random.randint(1, regions)
        towers.append([f"T{i:03d}", lat, lon, region_id])

    df = pd.DataFrame(towers, columns=["tower_id", "latitude", "longitude", "region_id"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✅ Saved: {output}")

def generate_support_tickets(num_users=500, output="data/raw/support_tickets.csv"):
    issue_types = ['Call Drop', 'Slow Internet', 'Billing', 'SIM Issue', 'No Signal', 'Other']
    tickets = []

    for user_id in range(1, num_users + 1):
        for _ in range(np.random.poisson(1.2)):
            ts = datetime(2025, 1, random.randint(1, 31), random.randint(0, 23), random.randint(0, 59))
            issue = random.choice(issue_types)
            resolved = np.random.choice([0, 1], p=[0.1, 0.9])
            tickets.append([f"U{user_id:04d}", ts.strftime("%Y-%m-%d %H:%M:%S"), issue, resolved])

    df = pd.DataFrame(tickets, columns=["user_id", "timestamp", "issue_type", "resolved"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✅ Saved: {output}")
    
def generate_user_profiles(num_users=500, num_regions=10, output="data/raw/user_profiles.csv"):
    fake = Faker()
    np.random.seed(42)

    user_ids = [f"U{i:04d}" for i in range(1, num_users + 1)]
    region_ids = [random.randint(1, num_regions) for _ in range(num_users)]
    ages = np.random.randint(18, 65, size=num_users)
    genders = np.random.choice(['M', 'F', 'O'], size=num_users, p=[0.48, 0.48, 0.04])
    signup_dates = [fake.date_between(start_date='-3y', end_date='-30d') for _ in range(num_users)]
    is_enterprise = np.random.choice([0, 1], size=num_users, p=[0.85, 0.15])

    df = pd.DataFrame({
        "user_id": user_ids,
        "age": ages,
        "gender": genders,
        "region_id": region_ids,
        "signup_date": signup_dates,
        "is_enterprise": is_enterprise
    })

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✅ user_profiles.csv saved to {output}")

if __name__ == "__main__":
    generate_synthetic_cdr()
    generate_anomaly_logs()
    generate_user_plan_history()
    generate_tower_locations()
    generate_support_tickets()
    generate_user_profiles()