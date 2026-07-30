import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import LABELED_DATA


def load_dataset(target_column: str):
    """
    Load the labeled observability dataset and separate
    input features from the selected prediction target.
    """

    print(f"\nLoading labeled dataset from:\n{LABELED_DATA}")

    if not LABELED_DATA.exists():
        raise FileNotFoundError(
            "Labeled dataset was not found.\n"
            f"Expected location:\n{LABELED_DATA}\n\n"
            "Run the labeling script first using:\n"
            "python -m src.data.labeling"
        )

    df = pd.read_csv(LABELED_DATA)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found.\n"
            f"Available target columns: "
            f"{[column for column in df.columns if column.startswith('Target_')]}"
        )

    # Columns that should not be used as model features
    columns_to_drop = [
        "Timestamp",
        "Service",
        "Incident",
        "Target_5min",
        "Target_10min",
        "Target_15min",
    ]

    feature_columns_to_drop = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    X = df.drop(columns=feature_columns_to_drop)
    y = df[target_column]

    # Ensure all model features are numeric
    non_numeric_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric feature columns found: "
            f"{non_numeric_columns}"
        )

    print(f"\nDataset shape: {df.shape}")
    print(f"Feature shape: {X.shape}")
    print(f"Target column: {target_column}")
    print(f"Positive samples: {int(y.sum())}")
    print(f"Negative samples: {int((y == 0).sum())}")

    return X, y


def prepare_train_test(
    target_column: str,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Load the dataset and create training and testing sets.
    """

    X, y = load_dataset(target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test