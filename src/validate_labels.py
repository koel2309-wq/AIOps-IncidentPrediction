from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("VALIDATING INCIDENT LABELS")
print("=" * 60)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "labeled_observability_metrics.csv"

RESULTS_DIR = BASE_DIR / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# --------------------------------------------------
# Select one service
# --------------------------------------------------

service = "Auth-Service"

service_df = df[df["Service"] == service].copy()

# --------------------------------------------------
# Find first incident
# --------------------------------------------------

incident_index = service_df[service_df["Incident"] == 1].index[0]

window = 40

plot_df = service_df.loc[
    incident_index - window:
    incident_index + 10
].copy()

print(f"Incident Index : {incident_index}")

# --------------------------------------------------
# Plot CPU
# --------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    plot_df["Timestamp"],
    plot_df["CPU"],
    label="CPU"
)

plt.scatter(
    plot_df[plot_df["Incident"] == 1]["Timestamp"],
    plot_df[plot_df["Incident"] == 1]["CPU"],
    color="red",
    label="Incident"
)

plt.fill_between(
    plot_df["Timestamp"],
    0,
    100,
    where=plot_df["Target_5min"] == 1,
    alpha=0.2,
    label="Prediction Window"
)

plt.title("CPU Behaviour Before Incident")

plt.xlabel("Time")

plt.ylabel("CPU Utilization (%)")

plt.legend()

plt.tight_layout()

plt.savefig(RESULTS_DIR / "cpu_prediction_window.png")

plt.close()

# --------------------------------------------------
# Plot Latency
# --------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    plot_df["Timestamp"],
    plot_df["Latency"],
    label="Latency"
)

plt.scatter(
    plot_df[plot_df["Incident"] == 1]["Timestamp"],
    plot_df[plot_df["Incident"] == 1]["Latency"],
    color="red",
    label="Incident"
)

plt.fill_between(
    plot_df["Timestamp"],
    0,
    plot_df["Latency"].max(),
    where=plot_df["Target_5min"] == 1,
    alpha=0.2,
    label="Prediction Window"
)

plt.title("Latency Before Incident")

plt.xlabel("Time")

plt.ylabel("Latency (ms)")

plt.legend()

plt.tight_layout()

plt.savefig(RESULTS_DIR / "latency_prediction_window.png")

plt.close()

# --------------------------------------------------
# Plot Error Rate
# --------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    plot_df["Timestamp"],
    plot_df["ErrorRate"],
    label="Error Rate"
)

plt.scatter(
    plot_df[plot_df["Incident"] == 1]["Timestamp"],
    plot_df[plot_df["Incident"] == 1]["ErrorRate"],
    color="red",
    label="Incident"
)

plt.fill_between(
    plot_df["Timestamp"],
    0,
    plot_df["ErrorRate"].max(),
    where=plot_df["Target_5min"] == 1,
    alpha=0.2,
    label="Prediction Window"
)

plt.title("Error Rate Before Incident")

plt.xlabel("Time")

plt.ylabel("Error Rate")

plt.legend()

plt.tight_layout()

plt.savefig(RESULTS_DIR / "error_prediction_window.png")

plt.close()

print("\nFigures saved in:")

print(RESULTS_DIR)

print("=" * 60)