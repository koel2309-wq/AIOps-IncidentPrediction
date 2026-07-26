import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "labeled_observability_metrics.csv"
file_path1 = BASE_DIR / "data" / "raw" / "observability_metrics.csv"

df = pd.read_csv(file_path1)
print(df["Incident"].sum())
