from pathlib import Path

import matplotlib.pyplot as plt

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

BASE_DIR = Path(__file__).resolve().parent.parent


def evaluate_model(model,
                   X_test,
                   y_test,
                   model_name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test,
                                predictions,
                                zero_division=0)

    recall = recall_score(y_test,
                          predictions,
                          zero_division=0)

    f1 = f1_score(y_test,
                  predictions,
                  zero_division=0)

    roc = roc_auc_score(y_test,
                        probabilities)

    print("\n" + "=" * 50)

    print(model_name)

    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC AUC  : {roc:.4f}")

    # ------------------------
    # Save confusion matrix
    # ------------------------

    cm_dir = BASE_DIR / "results" / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions)
    )

    disp.plot()

    plt.title(model_name)

    plt.savefig(cm_dir / f"{model_name}.png")

    plt.close()

    # ------------------------
    # Save ROC Curve
    # ------------------------

    roc_dir = BASE_DIR / "results" / "roc_curves"
    roc_dir.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(model_name)

    plt.savefig(roc_dir / f"{model_name}.png")

    plt.close()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC": roc
    }