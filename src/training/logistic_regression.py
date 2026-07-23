from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR
from src.training.train import train_model
from src.utils.preprocessing import prepare_train_test


def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL")
    print("=" * 60)

    # Use the same dataset split as the other models
    X_train, X_test, y_train, y_test = prepare_train_test(
        target_column="Target_5min"
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42
                )
            )
        ]
    )

    # train_model should return the fitted model
    trained_model = train_model(
        model=model,
        model_name="LogisticRegression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # --------------------------------------------------
    # Save predictions and probabilities for ROC comparison
    # --------------------------------------------------

    y_pred = trained_model.predict(X_test)

    y_probability = trained_model.predict_proba(
        X_test
    )[:, 1]

    predictions_dir = RESULTS_DIR / "predictions"

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_results = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Prediction": y_pred,
            "Probability": y_probability
        }
    )

    prediction_file = (
        predictions_dir
        / "LogisticRegression_predictions.csv"
    )

    prediction_results.to_csv(
        prediction_file,
        index=False
    )

    print(
        f"\nPrediction probabilities saved to:\n"
        f"{prediction_file}"
    )


if __name__ == "__main__":
    main()