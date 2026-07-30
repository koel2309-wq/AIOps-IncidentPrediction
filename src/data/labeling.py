import pandas as pd
from src.config import ENGINEERED_DATA, LABELED_DATA

print("=" * 60)
print("INCIDENT PREDICTION LABELING")
print("=" * 60)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print(f"\nLoading dataset:\n{ENGINEERED_DATA}")

df = pd.read_csv(ENGINEERED_DATA)

print(f"\nDataset Shape : {df.shape}")

# --------------------------------------------------
# Create Prediction Target Columns
# --------------------------------------------------

prediction_windows = [5, 10, 15]

for window in prediction_windows:
    df[f"Target_{window}min"] = 0

# --------------------------------------------------
# Process Each Service Independently
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
# Save Dataset
# --------------------------------------------------

LABELED_DATA.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(LABELED_DATA, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nPrediction Labels Created Successfully\n")

for window in prediction_windows:

    positives = df[f"Target_{window}min"].sum()

    print(f"Target_{window}min : {positives} positive samples")

print("\nSaved dataset to:")
print(LABELED_DATA)

print("\nFirst Five Rows:\n")
print(df.head())

print("=" * 60)