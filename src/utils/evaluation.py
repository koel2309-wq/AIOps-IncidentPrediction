from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from src.config import (
    METRICS_DIR,
    ROC_DIR,
    CONFUSION_DIR
)


def evaluate_model(
        model_name,
        y_true,
        y_pred,
        y_prob,
        model=None,
        X_test=None
):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc = roc_auc_score(
        y_true,
        y_prob
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")

    # ---------------------------------------
    # Save metrics
    # ---------------------------------------

    metrics = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc
    }])

    metrics.to_csv(
        METRICS_DIR / f"{model_name}.csv",
        index=False
    )

    # ---------------------------------------
    # Confusion Matrix
    # ---------------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred)
    )

    disp.plot()

    plt.title(model_name)

    plt.tight_layout()

    plt.savefig(
        CONFUSION_DIR / f"{model_name}.png"
    )

    plt.close()

    # ---------------------------------------
    # ROC Curve
    # ---------------------------------------

    if model is not None:

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_true
        )

        plt.title(model_name)

        plt.tight_layout()

        plt.savefig(
            ROC_DIR / f"{model_name}.png"
        )

        plt.close()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc
    }