import pandas as pd
import numpy as np
import os

np.random.seed(42)

SERVICES = [
    "Auth-Service",
    "Payment-Service",
    "Order-Service",
    "Inventory-Service",
    "Notification-Service"
]

OUTPUT_PATH = "../data/raw"

DAYS = 30

INTERVAL = "1min"


def generate_dataset():

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=DAYS * 24 * 60,
        freq=INTERVAL
    )

    records = []

    for service in SERVICES:

        cpu = np.random.normal(45, 8, len(timestamps))
        memory = np.random.normal(55, 6, len(timestamps))
        latency = np.random.normal(120, 15, len(timestamps))
        throughput = np.random.normal(500, 30, len(timestamps))
        error_rate = np.random.normal(0.2, 0.05, len(timestamps))

        incident = np.zeros(len(timestamps))

        # Inject failure every ~3 days
        for start in range(3000, len(timestamps), 4500):

            end = min(start + 30, len(timestamps))

            cpu[start:end] += np.linspace(20, 45, end - start)
            memory[start:end] += np.linspace(15, 30, end - start)

            latency[start:end] += np.linspace(100, 350, end - start)

            throughput[start:end] -= np.linspace(80, 220, end - start)

            error_rate[start:end] += np.linspace(1, 6, end - start)

            incident[end - 1] = 1

        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Service": service,
            "CPU": cpu.clip(10, 100),
            "Memory": memory.clip(20, 100),
            "Latency": latency.clip(20, 1000),
            "Throughput": throughput.clip(50, 1000),
            "ErrorRate": error_rate.clip(0, 100),
            "Incident": incident.astype(int)
        })

        records.append(df)

    final_df = pd.concat(records)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    final_df.to_csv(
        os.path.join(OUTPUT_PATH, "observability_metrics.csv"),
        index=False
    )

    print("=" * 50)
    print("Dataset Generated Successfully")
    print("=" * 50)
    print(final_df.head())
    print()
    print(f"Rows : {len(final_df)}")
    print(f"Services : {len(SERVICES)}")
    print(f"Incidents : {final_df['Incident'].sum()}")
    print("=" * 50)


if __name__ == "__main__":
    generate_dataset()