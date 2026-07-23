from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_observability_metrics.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

REQUIRED_MODEL_FILES = [
    MODELS_DIR / "LogisticRegression.joblib",
    MODELS_DIR / "RandomForest.joblib",
    MODELS_DIR / "XGBoost.joblib",
    MODELS_DIR / "LSTM_PyTorch.pt",
    MODELS_DIR / "LSTM_scaler.joblib",
]

REQUIRED_METRICS_FILES = {
    "Logistic Regression":
        RESULTS_DIR / "metrics" / "LogisticRegression.csv",
    "Random Forest":
        RESULTS_DIR / "metrics" / "RandomForest.csv",
    "XGBoost":
        RESULTS_DIR / "metrics" / "XGBoost.csv",
    "LSTM":
        RESULTS_DIR / "metrics" / "LSTM.csv",
}

REQUIRED_RESULT_FILES = [
    RESULTS_DIR / "confusion_matrices" / "LogisticRegression.png",
    RESULTS_DIR / "confusion_matrices" / "RandomForest.png",
    RESULTS_DIR / "confusion_matrices" / "XGBoost.png",
    RESULTS_DIR / "confusion_matrices" / "LSTM.png",
    RESULTS_DIR / "roc_curves" / "LogisticRegression.png",
    RESULTS_DIR / "roc_curves" / "RandomForest.png",
    RESULTS_DIR / "roc_curves" / "XGBoost.png",
    RESULTS_DIR / "roc_curves" / "LSTM.png",
    RESULTS_DIR / "training_history" / "LSTM_loss.png",
    RESULTS_DIR / "comparison" / "model_comparison.csv",
    RESULTS_DIR / "comparison" / "model_comparison.png",
]

REQUIRED_COLUMNS = [
    "Timestamp",
    "Service",
    "CPU",
    "Memory",
    "Latency",
    "Throughput",
    "ErrorRate",
    "Incident",
    "Target_5min",
    "Target_10min",
    "Target_15min",
]

METRIC_COLUMNS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
]


passed = 0
failed = 0
warnings = 0


def pass_check(message: str) -> None:
    global passed
    passed += 1
    print(f"PASS  | {message}")


def fail_check(message: str) -> None:
    global failed
    failed += 1
    print(f"FAIL  | {message}")


def warning_check(message: str) -> None:
    global warnings
    warnings += 1
    print(f"WARN  | {message}")


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def get_model_features(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return []


def validate_files() -> None:
    section("1. REQUIRED FILES")

    if DATA_FILE.exists():
        pass_check(f"Dataset exists: {DATA_FILE}")
    else:
        fail_check(f"Dataset missing: {DATA_FILE}")

    for model_file in REQUIRED_MODEL_FILES:
        if model_file.exists():
            pass_check(f"Model exists: {model_file.name}")
        else:
            fail_check(f"Model missing: {model_file}")

    for result_file in REQUIRED_RESULT_FILES:
        if result_file.exists() and result_file.stat().st_size > 0:
            pass_check(f"Result exists: {result_file.relative_to(PROJECT_ROOT)}")
        else:
            fail_check(f"Result missing or empty: {result_file}")


def validate_dataset() -> pd.DataFrame | None:
    section("2. DATASET VALIDATION")

    if not DATA_FILE.exists():
        fail_check("Cannot validate dataset because it is missing.")
        return None

    df = pd.read_csv(DATA_FILE)

    if len(df) == 216000:
        pass_check("Dataset contains 216,000 observations.")
    else:
        warning_check(
            f"Dataset contains {len(df):,} observations; expected approximately 216,000."
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if not missing_columns:
        pass_check("All required dataset columns are present.")
    else:
        fail_check(f"Missing dataset columns: {missing_columns}")

    service_count = df["Service"].nunique()

    if service_count == 5:
        pass_check("Dataset contains five microservices.")
    else:
        warning_check(f"Dataset contains {service_count} services.")

    incident_count = int(df["Incident"].sum())

    if incident_count > 0:
        pass_check(f"Dataset contains {incident_count} incident events.")
    else:
        fail_check("Dataset contains no incident events.")

    for target in [
        "Target_5min",
        "Target_10min",
        "Target_15min",
    ]:
        positive_count = int(df[target].sum())

        if positive_count > 0:
            pass_check(
                f"{target} contains {positive_count} positive observations."
            )
        else:
            fail_check(f"{target} contains no positive observations.")

    if (
        df["Target_5min"].sum()
        <= df["Target_10min"].sum()
        <= df["Target_15min"].sum()
    ):
        pass_check(
            "Prediction-window counts increase correctly from 5 to 15 minutes."
        )
    else:
        fail_check(
            "Prediction-window target counts are logically inconsistent."
        )

    invalid_target_values = {}

    for target in [
        "Incident",
        "Target_5min",
        "Target_10min",
        "Target_15min",
    ]:
        values = set(df[target].dropna().unique())

        if values.issubset({0, 1}):
            pass_check(f"{target} is binary.")
        else:
            invalid_target_values[target] = values

    if invalid_target_values:
        fail_check(
            f"Non-binary target values found: {invalid_target_values}"
        )

    numeric_columns = [
        "CPU",
        "Memory",
        "Latency",
        "Throughput",
        "ErrorRate",
    ]

    infinite_count = int(
        np.isinf(
            df[numeric_columns].to_numpy(dtype=float)
        ).sum()
    )

    if infinite_count == 0:
        pass_check("No infinite values exist in primary metrics.")
    else:
        fail_check(
            f"Found {infinite_count} infinite metric values."
        )

    return df


def validate_classical_models(df: pd.DataFrame | None) -> None:
    section("3. CLASSICAL MODEL VALIDATION")

    if df is None:
        fail_check("Dataset unavailable; model features cannot be checked.")
        return

    model_files = {
        "Logistic Regression":
            MODELS_DIR / "LogisticRegression.joblib",
        "Random Forest":
            MODELS_DIR / "RandomForest.joblib",
    }

    for model_name, model_path in model_files.items():
        if not model_path.exists():
            fail_check(f"{model_name} model is missing.")
            continue

        try:
            model = joblib.load(model_path)
            features = get_model_features(model)

            pass_check(f"{model_name} loads successfully.")

            if not features:
                warning_check(
                    f"{model_name} does not expose training feature names."
                )
                continue

            missing_features = [
                feature for feature in features
                if feature not in df.columns
            ]

            if not missing_features:
                pass_check(
                    f"{model_name} expected features exist in the dataset."
                )
            else:
                fail_check(
                    f"{model_name} missing features: {missing_features}"
                )

            if "Severity" in features:
                fail_check(
                    f"{model_name} uses Severity, creating possible target leakage."
                )
            else:
                pass_check(
                    f"{model_name} correctly excludes Severity."
                )

            forbidden_targets = {
                "Incident",
                "Target_5min",
                "Target_10min",
                "Target_15min",
            }.intersection(features)

            if forbidden_targets:
                fail_check(
                    f"{model_name} uses target columns as inputs: "
                    f"{sorted(forbidden_targets)}"
                )
            else:
                pass_check(
                    f"{model_name} contains no direct target leakage."
                )

        except Exception as error:
            fail_check(
                f"{model_name} could not be loaded: {error}"
            )


def validate_metrics() -> None:
    section("4. METRICS VALIDATION")

    collected_rows = []

    for model_name, metrics_file in REQUIRED_METRICS_FILES.items():
        if not metrics_file.exists():
            fail_check(f"{model_name} metrics file is missing.")
            continue

        try:
            metrics_df = pd.read_csv(metrics_file)

            if metrics_df.empty:
                fail_check(f"{model_name} metrics file is empty.")
                continue

            row = metrics_df.iloc[0]

            missing_columns = [
                metric for metric in METRIC_COLUMNS
                if metric not in metrics_df.columns
            ]

            if missing_columns:
                fail_check(
                    f"{model_name} metrics missing: {missing_columns}"
                )
                continue

            invalid_values = []

            for metric in METRIC_COLUMNS:
                value = float(row[metric])

                if not 0 <= value <= 1:
                    invalid_values.append(
                        f"{metric}={value}"
                    )

            if invalid_values:
                fail_check(
                    f"{model_name} has invalid metrics: {invalid_values}"
                )
            else:
                pass_check(
                    f"{model_name} metrics are all within [0, 1]."
                )

            if float(row["ROC_AUC"]) >= 0.95:
                warning_check(
                    f"{model_name} ROC-AUC is very high "
                    f"({float(row['ROC_AUC']):.4f}); explain the synthetic-data effect."
                )

            result = {
                "Model": model_name,
                **{
                    metric: float(row[metric])
                    for metric in METRIC_COLUMNS
                },
            }

            collected_rows.append(result)

            print(
                f"      {model_name}: "
                f"Precision={result['Precision']:.4f}, "
                f"Recall={result['Recall']:.4f}, "
                f"F1={result['F1']:.4f}, "
                f"ROC-AUC={result['ROC_AUC']:.4f}"
            )

        except Exception as error:
            fail_check(
                f"Could not validate {model_name} metrics: {error}"
            )

    if len(collected_rows) == 4:
        pass_check("Metrics exist for all four models.")

        comparison = pd.DataFrame(collected_rows)

        best_f1 = comparison.loc[
            comparison["F1"].idxmax()
        ]

        best_precision = comparison.loc[
            comparison["Precision"].idxmax()
        ]

        best_recall = comparison.loc[
            comparison["Recall"].idxmax()
        ]

        print("\nModel summary:")
        print(
            f"      Best F1: {best_f1['Model']} "
            f"({best_f1['F1']:.4f})"
        )
        print(
            f"      Best precision: {best_precision['Model']} "
            f"({best_precision['Precision']:.4f})"
        )
        print(
            f"      Best recall: {best_recall['Model']} "
            f"({best_recall['Recall']:.4f})"
        )


def validate_lstm() -> None:
    section("5. LSTM VALIDATION")

    metrics_file = REQUIRED_METRICS_FILES["LSTM"]

    if not metrics_file.exists():
        fail_check("LSTM metrics file is missing.")
        return

    metrics_df = pd.read_csv(metrics_file)

    if "Prediction_Threshold" not in metrics_df.columns:
        fail_check(
            "LSTM metrics do not contain the validation-selected threshold."
        )
    else:
        threshold = float(
            metrics_df.iloc[0]["Prediction_Threshold"]
        )

        if 0 < threshold < 1:
            pass_check(
                f"LSTM threshold is valid: {threshold:.6f}"
            )
        else:
            fail_check(
                f"LSTM threshold is invalid: {threshold}"
            )

    if (
        MODELS_DIR / "LSTM_PyTorch.pt"
    ).stat().st_size > 0:
        pass_check("LSTM checkpoint is non-empty.")

    if (
        MODELS_DIR / "LSTM_scaler.joblib"
    ).stat().st_size > 0:
        pass_check("LSTM scaler is non-empty.")


def validate_comparison() -> None:
    section("6. MODEL COMPARISON VALIDATION")

    comparison_file = (
        RESULTS_DIR
        / "comparison"
        / "model_comparison.csv"
    )

    if not comparison_file.exists():
        fail_check("Model comparison CSV is missing.")
        return

    comparison = pd.read_csv(
        comparison_file
    )

    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LSTM",
    }

    found_models = set(
        comparison["Model"]
    )

    missing_models = (
        expected_models - found_models
    )

    if not missing_models:
        pass_check(
            "Comparison table contains all four models."
        )
    else:
        fail_check(
            f"Comparison table missing models: {sorted(missing_models)}"
        )

    if len(comparison) == 4:
        pass_check("Comparison table contains exactly four rows.")
    else:
        warning_check(
            f"Comparison table contains {len(comparison)} rows."
        )


def print_manual_dashboard_checks() -> None:
    section("7. MANUAL DASHBOARD CHECKS")

    print(
        """
Complete these checks in the dashboard:

[ ] Logistic Regression loads without errors.
[ ] Random Forest loads without errors.
[ ] XGBoost loads through the isolated subprocess.
[ ] LSTM loads without errors.
[ ] At 20 minutes before an incident, Target_5min is normally 0.
[ ] At 5 minutes before an incident, Target_5min is 1.
[ ] At the incident timestamp, Incident is 1.
[ ] LSTM shows exactly ten historical observations.
[ ] Displayed CPU/Memory values agree with the final chart values.
[ ] Displayed latency/error values agree with the chart.
[ ] XGBoost can be selected repeatedly without a segmentation fault.
[ ] Switching between all four models does not terminate Streamlit.
"""
    )


def main() -> None:
    print("=" * 72)
    print("AIOPS INCIDENT PREDICTION — PROJECT VALIDATION")
    print("=" * 72)

    validate_files()
    dataframe = validate_dataset()
    validate_classical_models(dataframe)
    validate_metrics()
    validate_lstm()
    validate_comparison()
    print_manual_dashboard_checks()

    section("VALIDATION SUMMARY")

    print(f"Passed   : {passed}")
    print(f"Warnings : {warnings}")
    print(f"Failed   : {failed}")

    if failed == 0:
        print(
            "\nOVERALL RESULT: AUTOMATED VALIDATION PASSED"
        )
    else:
        print(
            "\nOVERALL RESULT: CORRECTIONS ARE REQUIRED"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()