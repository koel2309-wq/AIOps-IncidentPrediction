import matplotlib.pyplot as plt
import pandas as pd

from joblib import dump

from src.config import (
    FEATURE_IMPORTANCE_DIR,
    MODEL_DIR,
    RESULTS_DIR,
)

from src.utils.evaluation import evaluate_model


def train_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train, evaluate and save a classical machine learning model.

    The function also saves test-set predictions and probability
    scores so that ROC curves from multiple models can later be
    plotted in a single comparison figure.
    """

    print("=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Generate test predictions
    # --------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model_name} does not support predict_proba(). "
            "Probability scores are required for ROC analysis."
        )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # Evaluate model
    # --------------------------------------------------

    evaluate_model(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        model=model,
        X_test=X_test,
    )

    # --------------------------------------------------
    # Save test predictions for combined ROC comparison
    # --------------------------------------------------

    predictions_dir = (
        RESULTS_DIR / "predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_results = pd.DataFrame(
        {
            "Actual": pd.Series(y_test).reset_index(
                drop=True
            ),
            "Prediction": pd.Series(y_pred).reset_index(
                drop=True
            ),
            "Probability": pd.Series(y_prob).reset_index(
                drop=True
            ),
        }
    )

    prediction_file = (
        predictions_dir
        / f"{model_name}_predictions.csv"
    )

    prediction_results.to_csv(
        prediction_file,
        index=False,
    )

    print(
        f"\nSaved Predictions:\n{prediction_file}"
    )

    # --------------------------------------------------
    # Save feature importance when supported
    # --------------------------------------------------

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        ).sort_values(
            ascending=False
        )

        plt.figure(
            figsize=(10, 6)
        )

        importance.head(15).sort_values().plot(
            kind="barh"
        )

        plt.title(
            f"{model_name} Feature Importance"
        )

        plt.xlabel(
            "Importance"
        )

        plt.tight_layout()

        feature_importance_file = (
            FEATURE_IMPORTANCE_DIR
            / f"{model_name}.png"
        )

        plt.savefig(
            feature_importance_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\nSaved Feature Importance:\n"
            f"{feature_importance_file}"
        )

    # --------------------------------------------------
    # Save trained model
    # --------------------------------------------------

    model_file = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    dump(
        model,
        model_file,
    )

    print(
        f"\nSaved Model:\n{model_file}"
    )

    # Returning the fitted model is useful for future scripts,
    # although existing training files do not need to capture it.
    return model