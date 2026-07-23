import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Machine Learning-Based Incident Prediction
# Synthetic Observability Dataset Generator (Version 2.0)
# ---------------------------------------------------------

np.random.seed(42)

SERVICES = [
    "Auth-Service",
    "Payment-Service",
    "Order-Service",
    "Inventory-Service",
    "Notification-Service"
]

DAYS = 30
INTERVAL = "1min"

OUTPUT_PATH = "../../data/raw"


def generate_dataset():

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=DAYS * 24 * 60,
        freq=INTERVAL
    )

    records = []

    for service in SERVICES:

        print(f"Generating data for {service}...")

        # -------------------------------------------------
        # Normal Operating Behaviour
        # -------------------------------------------------

        cpu = np.random.normal(45, 8, len(timestamps))
        memory = np.random.normal(55, 6, len(timestamps))
        latency = np.random.normal(120, 15, len(timestamps))
        throughput = np.random.normal(500, 30, len(timestamps))
        error_rate = np.random.normal(0.20, 0.05, len(timestamps))

        incident = np.zeros(len(timestamps))
        severity = np.zeros(len(timestamps))

        # -------------------------------------------------
        # Inject Multiple Incident Types
        # -------------------------------------------------

        start = 2500

        while start < len(timestamps) - 100:

            duration = np.random.randint(15, 60)
            end = min(start + duration, len(timestamps))

            failure_type = np.random.choice(
                ["CPU", "MEMORY", "NETWORK"]
            )

            severity_level = np.random.choice(
                [1, 2, 3],
                p=[0.5, 0.3, 0.2]
            )

            multiplier = {
                1: 0.7,
                2: 1.0,
                3: 1.4
            }[severity_level]

            # ---------------- CPU Saturation ----------------

            if failure_type == "CPU":

                cpu[start:end] += np.linspace(
                    15, 45, end - start
                ) * multiplier

                memory[start:end] += np.linspace(
                    5, 15, end - start
                ) * multiplier

                latency[start:end] += np.linspace(
                    50, 250, end - start
                ) * multiplier

                throughput[start:end] -= np.linspace(
                    40, 120, end - start
                ) * multiplier

                error_rate[start:end] += np.linspace(
                    0.5, 4, end - start
                ) * multiplier

            # ---------------- Memory Leak ----------------

            elif failure_type == "MEMORY":

                memory[start:end] += np.linspace(
                    20, 40, end - start
                ) * multiplier

                cpu[start:end] += np.linspace(
                    5, 10, end - start
                ) * multiplier

                latency[start:end] += np.linspace(
                    40, 180, end - start
                ) * multiplier

                throughput[start:end] -= np.linspace(
                    20, 80, end - start
                ) * multiplier

                error_rate[start:end] += np.linspace(
                    0.5, 3, end - start
                ) * multiplier

            # ---------------- Network Failure ----------------

            else:

                latency[start:end] += np.linspace(
                    150, 450, end - start
                ) * multiplier

                throughput[start:end] -= np.linspace(
                    80, 250, end - start
                ) * multiplier

                error_rate[start:end] += np.linspace(
                    2, 8, end - start
                ) * multiplier

            incident[end - 1] = 1
            severity[end - 1] = severity_level

            # Random gap before next incident
            start += np.random.randint(1800, 3500)

        # -------------------------------------------------
        # Random False Spikes
        # -------------------------------------------------

        for _ in range(60):

            idx = np.random.randint(0, len(timestamps) - 1)

            cpu[idx] += np.random.randint(10, 20)
            latency[idx] += np.random.randint(100, 250)

            # No incident label

        # -------------------------------------------------
        # Missing Telemetry
        # -------------------------------------------------

        missing_cpu = np.random.choice(
            len(cpu),
            int(len(cpu) * 0.01),
            replace=False
        )

        cpu[missing_cpu] = np.nan

        missing_memory = np.random.choice(
            len(memory),
            int(len(memory) * 0.01),
            replace=False
        )

        memory[missing_memory] = np.nan

        # -------------------------------------------------
        # Build DataFrame
        # -------------------------------------------------

        df = pd.DataFrame({

            "Timestamp": timestamps,

            "Service": service,

            "CPU": np.clip(cpu, 10, 100),

            "Memory": np.clip(memory, 20, 100),

            "Latency": np.clip(latency, 20, 1000),

            "Throughput": np.clip(throughput, 50, 1000),

            "ErrorRate": np.clip(error_rate, 0, 100),

            "Incident": incident.astype(int),

            "Severity": severity.astype(int)

        })

        records.append(df)

    # -------------------------------------------------
    # Combine all services
    # -------------------------------------------------

    final_df = pd.concat(records, ignore_index=True)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_PATH,
        "observability_metrics.csv"
    )

    final_df.to_csv(output_file, index=False)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET GENERATED SUCCESSFULLY")
    print("=" * 60)

    print(final_df.head())

    print("\nRows :", len(final_df))
    print("Services :", len(SERVICES))
    print("Incidents :", int(final_df["Incident"].sum()))

    print("\nIncident Severity Distribution")

    print(
        final_df["Severity"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved to:")
    print(output_file)

    print("=" * 60)


if __name__ == "__main__":
    generate_dataset()