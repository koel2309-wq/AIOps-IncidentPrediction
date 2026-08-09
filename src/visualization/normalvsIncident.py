import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/labeled_observability_metrics.csv")

TARGET = "Target_5min"      # Change if needed

plt.figure(figsize=(8,5))

plt.hist(
    df[df[TARGET] == 0]["CPU_RollingMean"],
    bins=40,
    density=True,
    alpha=0.6,
    label="Normal"
)

plt.hist(
    df[df[TARGET] == 1]["CPU_RollingMean"],
    bins=30,
    density=True,
    alpha=0.7,
    label="Incident"
)

plt.xlabel("CPU Rolling Mean (%)")
plt.ylabel("Density")
plt.title("CPU Rolling Mean Distribution: Normal vs Incident")
plt.legend()

plt.tight_layout()
plt.savefig("results/cpu_distribution.png", dpi=300)
plt.show()