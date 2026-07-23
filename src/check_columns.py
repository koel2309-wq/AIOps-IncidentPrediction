import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "labeled_observability_metrics.csv"

df = pd.read_csv(file_path)

print(df.columns.tolist())