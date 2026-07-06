from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent


def load_dataset(target_column="Target_5min"):

    file = BASE_DIR / "data" / "processed" / "labeled_observability_metrics.csv"

    df = pd.read_csv(file)

    # --------------------------
    # Drop non-feature columns
    # --------------------------

    drop_columns = [
        "Timestamp",
        "Service",
        "Incident",
        "Target_5min",
        "Target_10min",
        "Target_15min"
    ]

    X = df.drop(columns=drop_columns)

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

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test