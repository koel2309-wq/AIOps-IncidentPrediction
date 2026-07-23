from pathlib import Path
import pandas as pd

print("=" * 60)
print("INCIDENT PREDICTION LABELING")
print("=" * 60)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "engineered_observability_metrics.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "labeled_observability_metrics.csv"

print(f"\nLoading dataset:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset Shape : {df.shape}")

# --------------------------------------------------
# Create prediction target columns
# --------------------------------------------------

prediction_windows = [5, 10, 15]

for window in prediction_windows:
    df[f"Target_{window}min"] = 0

# --------------------------------------------------
# Process each service independently
# --------------------------------------------------

services = df["Service"].unique()

for service in services:

    service_index = df[df["Service"] == service].index

    service_df = df.loc[service_index]

    incident_rows = service_df[service_df["Incident"] == 1].index

    for incident in incident_rows:

        for window in prediction_windows:

            start = max(service_index.min(), incident - window)

            df.loc[start:incident, f"Target_{window}min"] = 1

# --------------------------------------------------
# Save dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nPrediction Labels Created Successfully\n")

for window in prediction_windows:

    positives = df[f"Target_{window}min"].sum()

    print(f"Target_{window}min : {positives} positive samples")

print("\nSaved dataset to:")
print(OUTPUT_FILE)

print("\nFirst Five Rows:\n")
print(df.head())

print("=" * 60)