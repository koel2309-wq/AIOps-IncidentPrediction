from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.config import RESULTS_DIR, METRICS_DIR


# ============================================================
# Configuration
# ============================================================

SELECTION_METRIC = "F1"

MODEL_METRIC_FILES = {
    "Logistic Regression": METRICS_DIR / "LogisticRegression.csv",
    "Random Forest": METRICS_DIR / "RandomForest.csv",
    "XGBoost": METRICS_DIR / "XGBoost.csv",
    "LSTM": METRICS_DIR / "LSTM.csv",
}

# The script searches these possible locations for prediction files.
# Add or modify filenames here if your project uses different names.
PREDICTION_FILE_CANDIDATES = {
    "Logistic Regression": [
        RESULTS_DIR / "predictions" / "LogisticRegression_predictions.csv",
        RESULTS_DIR / "predictions" / "logistic_regression_predictions.csv",
        RESULTS_DIR / "LogisticRegression_predictions.csv",
        RESULTS_DIR / "logistic_regression_predictions.csv",
    ],
    "Random Forest": [
        RESULTS_DIR / "predictions" / "RandomForest_predictions.csv",
        RESULTS_DIR / "predictions" / "random_forest_predictions.csv",
        RESULTS_DIR / "RandomForest_predictions.csv",
        RESULTS_DIR / "random_forest_predictions.csv",
    ],
    "XGBoost": [
        RESULTS_DIR / "predictions" / "XGBoost_predictions.csv",
        RESULTS_DIR / "predictions" / "xgboost_predictions.csv",
        RESULTS_DIR / "XGBoost_predictions.csv",
        RESULTS_DIR / "xgboost_predictions.csv",
    ],
    "LSTM": [
        RESULTS_DIR / "predictions" / "LSTM_predictions.csv",
        RESULTS_DIR / "predictions" / "lstm_predictions.csv",
        RESULTS_DIR / "LSTM_predictions.csv",
        RESULTS_DIR / "lstm_predictions.csv",
    ],
}

TRUE_LABEL_COLUMNS = [
    "Actual",
    "actual",
    "y_true",
    "Y_True",
    "True_Label",
    "true_label",
    "Target",
    "target",
    "Label",
    "label",
]

PROBABILITY_COLUMNS = [
    "Probability",
    "probability",
    "Incident_Probability",
    "incident_probability",
    "y_probability",
    "Y_Probability",
    "y_prob",
    "Y_Prob",
    "y_score",
    "score",
]

REQUIRED_METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]


# ============================================================
# Helper Functions
# ============================================================

def find_existing_file(
    candidates: list[Path],
) -> Optional[Path]:
    """Return the first existing file from a list of candidates."""

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """Find the first matching column name."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def load_metric_results() -> pd.DataFrame:
    """
    Load the first row from every model metrics file and create
    a common comparison table.
    """

    rows = []

    for model_name, metric_file in MODEL_METRIC_FILES.items():
        if not metric_file.exists():
            print(f"Missing metrics file: {metric_file}")
            continue

        try:
            metric_df = pd.read_csv(metric_file)
        except Exception as error:
            print(
                f"Could not read metrics file for {model_name}: "
                f"{metric_file}"
            )
            print(f"Reason: {error}")
            continue

        if metric_df.empty:
            print(f"Metrics file is empty: {metric_file}")
            continue

        missing_columns = [
            column
            for column in REQUIRED_METRIC_COLUMNS
            if column not in metric_df.columns
        ]

        if missing_columns:
            print(
                f"Skipping {model_name}. Missing metric columns: "
                f"{missing_columns}"
            )
            continue

        row = metric_df.iloc[0].to_dict()
        row["Model"] = model_name
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No valid model metrics were found. "
            "Run the model evaluation scripts first."
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
        ]
    ].copy()

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    for column in numeric_columns:
        comparison[column] = pd.to_numeric(
            comparison[column],
            errors="coerce",
        )

    comparison = comparison.dropna(
        subset=numeric_columns
    )

    if comparison.empty:
        raise RuntimeError(
            "The metric files were found, but no valid numeric "
            "metric values were available."
        )

    if SELECTION_METRIC not in comparison.columns:
        raise ValueError(
            f"Invalid selection metric: {SELECTION_METRIC}"
        )

    comparison = comparison.sort_values(
        by=SELECTION_METRIC,
        ascending=False,
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "Rank",
        range(1, len(comparison) + 1),
    )

    return comparison


def save_comparison_table(
    comparison: pd.DataFrame,
    comparison_dir: Path,
) -> Path:
    """Save the consolidated comparison table."""

    output_path = comparison_dir / "model_comparison.csv"

    comparison.to_csv(
        output_path,
        index=False,
    )

    return output_path


def create_full_metric_chart(
    comparison: pd.DataFrame,
    comparison_dir: Path,
) -> Path:
    """
    Create the complete grouped bar chart containing all metrics.
    """

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    plot_df = comparison.set_index("Model")[metrics]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "Comparison of Machine Learning Models"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(
        axis="x",
        rotation=0,
    )
    ax.grid(
        axis="y",
        alpha=0.4,
    )
    ax.legend(
        title="Metric",
        loc="lower center",
        ncol=5,
    )

    plt.tight_layout()

    output_path = (
        comparison_dir / "model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def create_focused_metric_chart(
    comparison: pd.DataFrame,
    comparison_dir: Path,
) -> Path:
    """
    Create a focused chart containing the metrics most useful
    for an imbalanced classification problem.
    """

    metrics = [
        "Precision",
        "Recall",
        "F1",
    ]

    plot_df = comparison.set_index("Model")[metrics]

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Precision, Recall and F1-Score Comparison"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(
        axis="x",
        rotation=0,
    )
    ax.grid(
        axis="y",
        alpha=0.4,
    )
    ax.legend(
        title="Metric",
        loc="upper right",
    )

    plt.tight_layout()

    output_path = (
        comparison_dir / "model_comparison_focus.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def load_roc_data(
    model_name: str,
) -> Optional[tuple[pd.Series, pd.Series, Path]]:
    """
    Load true labels and probability scores for one model.

    Returns:
        y_true, y_probability, source_file

    Returns None when no valid prediction file is available.
    """

    prediction_file = find_existing_file(
        PREDICTION_FILE_CANDIDATES[model_name]
    )

    if prediction_file is None:
        print(
            f"No prediction probability file found for "
            f"{model_name}."
        )
        return None

    try:
        prediction_df = pd.read_csv(prediction_file)
    except Exception as error:
        print(
            f"Could not read prediction file for {model_name}: "
            f"{prediction_file}"
        )
        print(f"Reason: {error}")
        return None

    true_column = find_column(
        prediction_df,
        TRUE_LABEL_COLUMNS,
    )

    probability_column = find_column(
        prediction_df,
        PROBABILITY_COLUMNS,
    )

    if true_column is None or probability_column is None:
        print(
            f"Prediction file for {model_name} does not contain "
            f"the required true-label and probability columns."
        )
        print(
            f"Available columns: "
            f"{list(prediction_df.columns)}"
        )
        return None

    roc_df = prediction_df[
        [
            true_column,
            probability_column,
        ]
    ].copy()

    roc_df[true_column] = pd.to_numeric(
        roc_df[true_column],
        errors="coerce",
    )

    roc_df[probability_column] = pd.to_numeric(
        roc_df[probability_column],
        errors="coerce",
    )

    roc_df = roc_df.dropna()

    if roc_df.empty:
        print(
            f"No valid ROC data found for {model_name}."
        )
        return None

    unique_labels = sorted(
        roc_df[true_column].unique().tolist()
    )

    if len(unique_labels) != 2:
        print(
            f"ROC curve requires two classes. "
            f"{model_name} contains: {unique_labels}"
        )
        return None

    y_true = roc_df[true_column].astype(int)
    y_probability = roc_df[probability_column].astype(float)

    return (
        y_true,
        y_probability,
        prediction_file,
    )


def create_roc_comparison(
    comparison_dir: Path,
) -> Optional[Path]:
    """
    Plot all available model ROC curves in one graph.

    The output is skipped if no valid prediction probability
    files are available.
    """

    plt.figure(
        figsize=(8, 6)
    )

    plotted_models = 0

    for model_name in MODEL_METRIC_FILES:
        roc_data = load_roc_data(
            model_name
        )

        if roc_data is None:
            continue

        y_true, y_probability, source_file = roc_data

        try:
            false_positive_rate, true_positive_rate, _ = (
                roc_curve(
                    y_true,
                    y_probability,
                )
            )

            roc_auc = auc(
                false_positive_rate,
                true_positive_rate,
            )
        except Exception as error:
            print(
                f"Could not calculate ROC curve for "
                f"{model_name}: {error}"
            )
            continue

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=(
                f"{model_name} "
                f"(AUC = {roc_auc:.4f})"
            ),
        )

        print(
            f"ROC data loaded for {model_name}: "
            f"{source_file}"
        )

        plotted_models += 1

    if plotted_models == 0:
        plt.close()

        print(
            "\nROC comparison was not generated because no valid "
            "prediction probability files were found."
        )
        print(
            "Each prediction CSV should contain one true-label "
            "column and one probability column."
        )

        return None

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier",
    )

    plt.title(
        "ROC Curve Comparison"
    )
    plt.xlabel(
        "False Positive Rate"
    )
    plt.ylabel(
        "True Positive Rate"
    )
    plt.xlim(
        0,
        1,
    )
    plt.ylim(
        0,
        1.02,
    )
    plt.grid(
        alpha=0.3,
    )
    plt.legend(
        loc="lower right",
    )
    plt.tight_layout()

    output_path = (
        comparison_dir / "roc_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def print_summary(
    comparison: pd.DataFrame,
) -> None:
    """Print the comparison and highest-ranked model."""

    print("\nModel Comparison\n")

    printable = comparison.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    printable[metric_columns] = printable[
        metric_columns
    ].round(4)

    print(
        printable.to_string(
            index=False
        )
    )

    highest_ranked = comparison.iloc[0]

    print("\n" + "-" * 60)
    print("Highest-Ranked Model")
    print("-" * 60)
    print(
        f"Model            : "
        f"{highest_ranked['Model']}"
    )
    print(
        f"Ranking Metric   : "
        f"{SELECTION_METRIC}"
    )
    print(
        f"{SELECTION_METRIC} Score      : "
        f"{highest_ranked[SELECTION_METRIC]:.4f}"
    )

    print(
        "\nNote: this ranking is reported for comparative "
        "analysis only. The application does not automatically "
        "select or deploy a best model."
    )


# ============================================================
# Main Program
# ============================================================

def main() -> None:
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison_dir = (
        RESULTS_DIR / "comparison"
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = load_metric_results()

    comparison_csv = save_comparison_table(
        comparison,
        comparison_dir,
    )

    full_chart = create_full_metric_chart(
        comparison,
        comparison_dir,
    )

    focused_chart = create_focused_metric_chart(
        comparison,
        comparison_dir,
    )

    roc_chart = create_roc_comparison(
        comparison_dir
    )

    print_summary(
        comparison
    )

    print("\nSaved outputs:")
    print(comparison_csv)
    print(full_chart)
    print(focused_chart)

    if roc_chart is not None:
        print(roc_chart)

    print("=" * 60)


if __name__ == "__main__":
    main()