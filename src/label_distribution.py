import pandas as pd

dataset_path = "src/data/processed/labeled_observability_metrics.csv"

df = pd.read_csv(dataset_path)

print(f"Total rows: {len(df):,}\n")

for column in ["Target_5min", "Target_10min", "Target_15min"]:
    positive = int((df[column] == 1).sum())
    negative = int((df[column] == 0).sum())

    print("-" * 35)
    print(column)
    print(f"Positive samples: {positive:,}")
    print(f"Negative samples: {negative:,}")
    print(f"Total samples:    {positive + negative:,}")