import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent


def load_dataset(target_column="Target_5min"):

    df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "labeled_observability_metrics.csv"
    )

    # Remove columns that should not be used as model inputs
    X = df.drop(columns=[
        "Timestamp",
        "Service",
        "Incident",
        "Severity",
        "Target_5min",
        "Target_10min",
        "Target_15min"
    ])

    y = df[target_column]

    return X, y


def prepare_train_test(target_column="Target_5min"):

    X, y = load_dataset(target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test