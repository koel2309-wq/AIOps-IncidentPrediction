from pathlib import Path
import pandas as pd
from src.config import RAW_DATA, ENGINEERED_DATA

print("=" * 60)
print("FEATURE ENGINEERING STARTED")
print("=" * 60)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "engineered_observability_metrics.csv"

# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print(f"\nLoading dataset from:\n{RAW_DATA}\n")

df = pd.read_csv(RAW_DATA)

print(f"Dataset Shape: {df.shape}")

# --------------------------------------------------
# Convert Timestamp
# --------------------------------------------------

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort data by Service and Timestamp
df = df.sort_values(["Service", "Timestamp"])

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

window_size = 5

metrics = [
    "CPU",
    "Memory",
    "Latency",
    "Throughput",
    "ErrorRate"
]

print("\nCreating engineered features...\n")

for metric in metrics:

    print(f"Processing {metric}...")

    # Rolling Mean
    df[f"{metric}_RollingMean"] = (
        df.groupby("Service")[metric]
        .transform(lambda x: x.rolling(window_size, min_periods=1).mean())
    )

    # Rolling Standard Deviation
    df[f"{metric}_RollingStd"] = (
        df.groupby("Service")[metric]
        .transform(lambda x: x.rolling(window_size, min_periods=1).std())
    )

    # Difference from previous observation
    df[f"{metric}_Delta"] = (
        df.groupby("Service")[metric]
        .diff()
    )

    # Previous value (Lag-1)
    df[f"{metric}_Lag1"] = (
        df.groupby("Service")[metric]
        .shift(1)
    )

# --------------------------------------------------
# Fill Missing Values
# --------------------------------------------------

df.fillna(0, inplace=True)

# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

df.to_csv(ENGINEERED_DATA, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nOriginal Columns : 8")
print(f"Engineered Columns : {len(df.columns)}")

print(f"\nDataset Shape : {df.shape}")

print(f"\nSaved to:\n{OUTPUT_FILE}")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nEngineered Features Added:\n")

for metric in metrics:
    print(f"• {metric}_RollingMean")
    print(f"• {metric}_RollingStd")
    print(f"• {metric}_Delta")
    print(f"• {metric}_Lag1")
    print()

print("=" * 60)