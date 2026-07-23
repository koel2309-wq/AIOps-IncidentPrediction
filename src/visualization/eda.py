import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folder
FIGURE_PATH = "../../results/figures"
os.makedirs(FIGURE_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv("../../data/raw/observability_metrics.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nIncident Distribution:")
print(df["Incident"].value_counts())

# -------------------------
# Histograms
# -------------------------

metrics = ["CPU", "Memory", "Latency", "Throughput", "ErrorRate"]

for metric in metrics:

    plt.figure(figsize=(8,5))
    plt.hist(df[metric], bins=40)
    plt.title(f"{metric} Distribution")
    plt.xlabel(metric)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/{metric}_distribution.png")
    plt.close()

print("\nMetric distribution plots saved.")

# -------------------------
# Incident Distribution
# -------------------------

plt.figure(figsize=(6,4))
df["Incident"].value_counts().plot(kind="bar")

plt.title("Incident Distribution")
plt.xlabel("Incident")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(f"{FIGURE_PATH}/incident_distribution.png")
plt.close()

print("Incident distribution saved.")

print("\nEDA Completed Successfully.")