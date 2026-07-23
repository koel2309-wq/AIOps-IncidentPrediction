from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "XGBoost.joblib"


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError(
            "Expected one JSON argument containing model features."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_FILE}"
        )

    feature_values = json.loads(sys.argv[1])

    model = joblib.load(MODEL_FILE)

    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        feature_columns = list(
            model.get_booster().feature_names
        )
    else:
        feature_columns = list(feature_values.keys())

    missing_features = [
        column
        for column in feature_columns
        if column not in feature_values
    ]

    if missing_features:
        raise ValueError(
            f"Missing XGBoost features: {missing_features}"
        )

    model_input = pd.DataFrame(
        [
            {
                column: feature_values[column]
                for column in feature_columns
            }
        ],
        columns=feature_columns,
    )

    prediction = int(model.predict(model_input)[0])

    probability = float(
        model.predict_proba(model_input)[0, 1]
    )

    print(
        json.dumps(
            {
                "prediction": prediction,
                "probability": probability,
                "feature_columns": feature_columns,
            }
        )
    )


if __name__ == "__main__":
    main()