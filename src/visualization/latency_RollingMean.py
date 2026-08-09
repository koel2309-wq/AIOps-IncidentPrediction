import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/labeled_observability_metrics.csv")

TARGET = "Target_5min"      # Change if required

plt.figure(figsize=(8,5))

# Normal observations
plt.hist(
    df[df[TARGET] == 0]["Latency_RollingMean"],
    bins=40,
    density=True,
    alpha=0.6,
    label="Normal"
)

# Incident observations
plt.hist(
    df[df[TARGET] == 1]["Latency_RollingMean"],
    bins=30,
    density=True,
    alpha=0.7,
    label="Incident"
)

plt.xlabel("Latency Rolling Mean (ms)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Latency Rolling Mean Distribution: Normal vs Incident", fontsize=14)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/Latency_RollingMean_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()