from __future__ import annotations

import copy
import random
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CONFUSION_DIR,
    LABELED_DATA,
    METRICS_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    ROC_DIR,
)


# =========================================================
# Configuration
# =========================================================

RANDOM_STATE = 42
TARGET_COLUMN = "Target_5min"

SEQUENCE_LENGTH = 10

# The final 20% of each service is retained as the untouched test period.
# The preceding 80% is divided chronologically into training and validation.
TRAIN_TEST_SPLIT_RATIO = 0.80
VALIDATION_FRACTION_OF_TRAINING_PERIOD = 0.20

EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
MINIMUM_VALIDATION_IMPROVEMENT = 1e-5

BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Retain all positive training sequences and at most five negative sequences
# for every positive sequence. Validation and test data are not undersampled.
NEGATIVE_TO_POSITIVE_RATIO = 5

FEATURE_COLUMNS = [
    "CPU",
    "Memory",
    "Latency",
    "Throughput",
    "ErrorRate",
    "CPU_RollingMean",
    "Memory_RollingMean",
    "Latency_RollingMean",
    "Throughput_RollingMean",
    "ErrorRate_RollingMean",
]


# =========================================================
# Reproducibility
# =========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# =========================================================
# Output Directories
# =========================================================

def ensure_output_directories() -> None:
    """Create all directories required by this module."""

    required_directories = [
        MODEL_DIR,
        METRICS_DIR,
        ROC_DIR,
        CONFUSION_DIR,
        RESULTS_DIR / "predictions",
        RESULTS_DIR / "training_history",
    ]

    for directory in required_directories:
        directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# Device Selection
# =========================================================

def get_device() -> torch.device:
    """Use Apple Metal acceleration when available; otherwise use CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# Sequence Creation
# =========================================================

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ordered observations into fixed-length sequences.

    The target associated with each sequence is the label at the final
    timestamp in that sequence.
    """

    sequence_count = len(features) - sequence_length + 1

    if sequence_count <= 0:
        return (
            np.empty(
                (0, sequence_length, features.shape[1]),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.float32),
        )

    X_sequences = np.empty(
        (sequence_count, sequence_length, features.shape[1]),
        dtype=np.float32,
    )

    y_sequences = np.empty(
        sequence_count,
        dtype=np.float32,
    )

    for sequence_index in range(sequence_count):
        end_index = sequence_index + sequence_length

        X_sequences[sequence_index] = features[
            sequence_index:end_index
        ]

        y_sequences[sequence_index] = targets[
            end_index - 1
        ]

    return X_sequences, y_sequences


# =========================================================
# Training Undersampling
# =========================================================

def undersample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retain all positive training sequences and randomly sample negatives.

    Only the training subset is undersampled. Validation and test subsets
    retain their natural class distributions.
    """

    positive_indices = np.where(y_train == 1)[0]
    negative_indices = np.where(y_train == 0)[0]

    if len(positive_indices) == 0:
        raise ValueError(
            "No positive incident sequences exist in the training data."
        )

    negative_count = min(
        len(negative_indices),
        len(positive_indices) * ratio,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    sampled_negative_indices = rng.choice(
        negative_indices,
        size=negative_count,
        replace=False,
    )

    selected_indices = np.concatenate(
        [positive_indices, sampled_negative_indices]
    )

    rng.shuffle(selected_indices)

    return (
        X_train[selected_indices],
        y_train[selected_indices],
    )


# =========================================================
# Dataset Preparation
# =========================================================

def prepare_lstm_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """
    Prepare chronological training, validation and test sequences.

    For each service:
      1. The first 80% forms the development period.
      2. The final 20% forms the untouched test period.
      3. The final 20% of the development period forms validation data.
      4. The remaining development period forms training data.

    The scaler is fitted only on the actual training observations.
    """

    print(f"\nLoading data from:\n{LABELED_DATA}")

    if not LABELED_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {LABELED_DATA}"
        )

    df = pd.read_csv(LABELED_DATA)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    required_columns = [
        "Timestamp",
        "Service",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.sort_values(
        ["Service", "Timestamp"]
    ).reset_index(drop=True)

    # Fill missing telemetry independently for each service.
    df[FEATURE_COLUMNS] = (
        df.groupby("Service")[FEATURE_COLUMNS]
        .transform(lambda group: group.ffill().bfill())
    )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    training_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for service, service_df in df.groupby(
        "Service",
        sort=False,
    ):
        service_df = service_df.sort_values(
            "Timestamp"
        ).reset_index(drop=True)

        development_end = int(
            len(service_df) * TRAIN_TEST_SPLIT_RATIO
        )

        development_df = service_df.iloc[
            :development_end
        ].copy()

        service_test_df = service_df.iloc[
            development_end:
        ].copy()

        validation_size = max(
            SEQUENCE_LENGTH,
            int(
                len(development_df)
                * VALIDATION_FRACTION_OF_TRAINING_PERIOD
            ),
        )

        if len(development_df) <= validation_size:
            raise ValueError(
                f"Service '{service}' does not contain enough observations "
                "for chronological training and validation sequences."
            )

        service_training_df = development_df.iloc[
            :-validation_size
        ].copy()

        service_validation_df = development_df.iloc[
            -validation_size:
        ].copy()

        training_frames.append(service_training_df)
        validation_frames.append(service_validation_df)
        test_frames.append(service_test_df)

    training_df = pd.concat(
        training_frames,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_frames,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_frames,
        ignore_index=True,
    )

    scaler = StandardScaler()

    # Fit only on the true training period to avoid leakage.
    scaler.fit(
        training_df[FEATURE_COLUMNS]
    )

    X_training_parts: list[np.ndarray] = []
    y_training_parts: list[np.ndarray] = []

    X_validation_parts: list[np.ndarray] = []
    y_validation_parts: list[np.ndarray] = []

    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []

    for service in df["Service"].unique():
        service_training = training_df[
            training_df["Service"] == service
        ].sort_values("Timestamp")

        service_validation = validation_df[
            validation_df["Service"] == service
        ].sort_values("Timestamp")

        service_test = test_df[
            test_df["Service"] == service
        ].sort_values("Timestamp")

        training_features = scaler.transform(
            service_training[FEATURE_COLUMNS]
        ).astype(np.float32)

        validation_features = scaler.transform(
            service_validation[FEATURE_COLUMNS]
        ).astype(np.float32)

        test_features = scaler.transform(
            service_test[FEATURE_COLUMNS]
        ).astype(np.float32)

        training_targets = (
            service_training[TARGET_COLUMN]
            .astype(np.float32)
            .to_numpy()
        )

        validation_targets = (
            service_validation[TARGET_COLUMN]
            .astype(np.float32)
            .to_numpy()
        )

        test_targets = (
            service_test[TARGET_COLUMN]
            .astype(np.float32)
            .to_numpy()
        )

        X_service_training, y_service_training = create_sequences(
            training_features,
            training_targets,
            SEQUENCE_LENGTH,
        )

        X_service_validation, y_service_validation = create_sequences(
            validation_features,
            validation_targets,
            SEQUENCE_LENGTH,
        )

        X_service_test, y_service_test = create_sequences(
            test_features,
            test_targets,
            SEQUENCE_LENGTH,
        )

        if len(X_service_training) == 0:
            raise ValueError(
                f"No training sequences were created for service '{service}'."
            )

        if len(X_service_validation) == 0:
            raise ValueError(
                f"No validation sequences were created for service '{service}'."
            )

        if len(X_service_test) == 0:
            raise ValueError(
                f"No test sequences were created for service '{service}'."
            )

        X_training_parts.append(X_service_training)
        y_training_parts.append(y_service_training)

        X_validation_parts.append(X_service_validation)
        y_validation_parts.append(y_service_validation)

        X_test_parts.append(X_service_test)
        y_test_parts.append(y_service_test)

    return (
        np.concatenate(X_training_parts),
        np.concatenate(X_validation_parts),
        np.concatenate(X_test_parts),
        np.concatenate(y_training_parts),
        np.concatenate(y_validation_parts),
        np.concatenate(y_test_parts),
        scaler,
    )


# =========================================================
# PyTorch LSTM Model
# =========================================================

class IncidentLSTM(nn.Module):
    """LSTM classifier for incident prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.20)

        self.hidden_layer = nn.Linear(
            hidden_size,
            8,
        )

        self.output_layer = nn.Linear(
            8,
            1,
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        sequence_output, _ = self.lstm(inputs)

        final_time_step = sequence_output[:, -1, :]

        hidden = self.dropout(final_time_step)

        hidden = self.relu(
            self.hidden_layer(hidden)
        )

        logits = self.output_layer(hidden)

        return logits.squeeze(1)


# =========================================================
# Training
# =========================================================

def train_lstm(
    model: IncidentLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[float], float, int]:
    """
    Train the LSTM with validation-loss early stopping.

    The model state associated with the lowest validation loss is restored
    before the function returns.
    """

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    training_losses: list[float] = []
    validation_losses: list[float] = []

    best_validation_loss = float("inf")
    best_model_state: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()

        epoch_training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            epoch_training_loss += (
                loss.item() * X_batch.size(0)
            )

        average_training_loss = (
            epoch_training_loss
            / len(train_loader.dataset)
        )

        model.eval()

        epoch_validation_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch,
                )

                epoch_validation_loss += (
                    loss.item() * X_batch.size(0)
                )

        average_validation_loss = (
            epoch_validation_loss
            / len(validation_loader.dataset)
        )

        training_losses.append(
            average_training_loss
        )

        validation_losses.append(
            average_validation_loss
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Training Loss: {average_training_loss:.4f} - "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        validation_improved = (
            average_validation_loss
            < best_validation_loss - MINIMUM_VALIDATION_IMPROVEMENT
        )

        if validation_improved:
            best_validation_loss = average_validation_loss
            best_model_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "\nEarly stopping activated after "
                    f"{epoch + 1} epochs."
                )
                break

    training_time = (
        time.perf_counter()
        - training_start
    )

    if best_model_state is None:
        raise RuntimeError(
            "Training completed without producing a valid model state."
        )

    model.load_state_dict(best_model_state)

    print(
        f"\nRestored model from epoch {best_epoch} "
        f"with validation loss {best_validation_loss:.6f}."
    )

    return (
        training_losses,
        validation_losses,
        training_time,
        best_epoch,
    )


# =========================================================
# Probability Prediction
# =========================================================

def predict_probabilities(
    model: IncidentLSTM,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probabilities for every sequence in a data loader."""

    model.eval()

    probability_batches: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probabilities = torch.sigmoid(
                logits
            )

            probability_batches.append(
                probabilities.detach().cpu().numpy()
            )

    if not probability_batches:
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(
        probability_batches
    ).reshape(-1)


# =========================================================
# Validation-Based Threshold Selection
# =========================================================

def select_best_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
) -> float:
    """
    Select the probability threshold that maximises validation F1-score.

    Threshold selection uses the untouched validation set, which retains
    its natural class distribution.
    """

    y_validation_int = y_validation.astype(int).reshape(-1)
    validation_probabilities = validation_probabilities.reshape(-1)

    unique_classes = np.unique(y_validation_int)

    if len(unique_classes) < 2:
        print(
            "\nValidation data contains only one class. "
            "Using the default threshold of 0.5."
        )
        return 0.5

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_validation_int,
            validation_probabilities,
        )
    )

    if len(thresholds) == 0:
        print(
            "\nNo thresholds were generated. "
            "Using the default threshold of 0.5."
        )
        return 0.5

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (
            precision_values[:-1]
            + recall_values[:-1]
            + 1e-10
        )
    )

    best_index = int(
        np.nanargmax(f1_values)
    )

    best_threshold = float(
        thresholds[best_index]
    )

    print("\nValidation Threshold Selection")
    print(f"Best threshold       : {best_threshold:.6f}")
    print(
        f"Validation precision : "
        f"{precision_values[best_index]:.4f}"
    )
    print(
        f"Validation recall    : "
        f"{recall_values[best_index]:.4f}"
    )
    print(
        f"Validation F1        : "
        f"{f1_values[best_index]:.4f}"
    )

    return best_threshold


# =========================================================
# Evaluation
# =========================================================

def evaluate_lstm(
    model: IncidentLSTM,
    test_loader: DataLoader,
    y_test: np.ndarray,
    device: torch.device,
    training_time: float,
    prediction_threshold: float,
    best_epoch: int,
) -> dict[str, float | int]:
    """
    Evaluate the trained model and save all result artefacts.

    Metrics, saved predictions and the confusion matrix are generated from
    the same prediction array, ensuring internal consistency.
    """

    inference_start = time.perf_counter()

    probabilities = predict_probabilities(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    predictions = (
        probabilities >= prediction_threshold
    ).astype(int)

    y_test_int = y_test.astype(int).reshape(-1)

    if len(probabilities) != len(y_test_int):
        raise ValueError(
            "The number of predicted probabilities does not match "
            "the number of test labels."
        )

    predictions_dir = RESULTS_DIR / "predictions"

    prediction_results = pd.DataFrame(
        {
            "Actual": y_test_int,
            "Prediction": predictions,
            "Probability": probabilities,
            "Threshold": prediction_threshold,
        }
    )

    prediction_file = predictions_dir / "LSTM_predictions.csv"

    prediction_results.to_csv(
        prediction_file,
        index=False,
    )

    print(f"\nSaved Predictions:\n{prediction_file}")

    accuracy = accuracy_score(
        y_test_int,
        predictions,
    )

    precision = precision_score(
        y_test_int,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test_int,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test_int,
        predictions,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            y_test_int,
            probabilities,
        )
    except ValueError:
        roc_auc = float("nan")

    matrix = confusion_matrix(
        y_test_int,
        predictions,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    print("\n" + "=" * 50)
    print("PyTorch LSTM")
    print("=" * 50)

    print(
        f"Selected Threshold: "
        f"{prediction_threshold:.6f}"
    )

    print(
        f"Predicted Positives: "
        f"{int(predictions.sum())}"
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print(f"\nTrue Negatives : {true_negative}")
    print(f"False Positives: {false_positive}")
    print(f"False Negatives: {false_negative}")
    print(f"True Positives : {true_positive}")

    print(f"\nBest Epoch    : {best_epoch}")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.2f} seconds")

    metrics: dict[str, float | int] = {
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "ROC_AUC": float(roc_auc),
        "Prediction_Threshold": float(prediction_threshold),
        "Best_Epoch": int(best_epoch),
        "Training_Time_Seconds": float(training_time),
        "Inference_Time_Seconds": float(inference_time),
    }

    metrics_file = METRICS_DIR / "LSTM.csv"

    pd.DataFrame([metrics]).to_csv(
        metrics_file,
        index=False,
    )

    print(f"\nSaved Metrics:\n{metrics_file}")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Incident"],
    )

    display.plot(
        values_format="d"
    )

    plt.title("PyTorch LSTM Confusion Matrix")
    plt.tight_layout()

    confusion_file = CONFUSION_DIR / "LSTM.png"

    plt.savefig(
        confusion_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved Confusion Matrix:\n{confusion_file}")

    RocCurveDisplay.from_predictions(
        y_test_int,
        probabilities,
        name="LSTM",
    )

    plt.title("PyTorch LSTM ROC Curve")
    plt.tight_layout()

    roc_file = ROC_DIR / "LSTM.png"

    plt.savefig(
        roc_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved ROC Curve:\n{roc_file}")

    return metrics


# =========================================================
# Training History
# =========================================================

def save_training_history(
    training_losses: list[float],
    validation_losses: list[float],
    best_epoch: int,
) -> None:
    """Save the training and validation loss curves."""

    history_directory = RESULTS_DIR / "training_history"
    completed_epochs = len(training_losses)

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, completed_epochs + 1),
        training_losses,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        range(1, completed_epochs + 1),
        validation_losses,
        marker="o",
        label="Validation Loss",
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best Epoch ({best_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("PyTorch LSTM Training History")
    plt.legend()
    plt.tight_layout()

    history_file = history_directory / "LSTM_loss.png"

    plt.savefig(
        history_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved Training History:\n{history_file}")


# =========================================================
# Model Checkpoint
# =========================================================

def save_model_checkpoint(
    model: IncidentLSTM,
    scaler: StandardScaler,
    prediction_threshold: float,
    best_epoch: int,
) -> None:
    """
    Save the model, preprocessing metadata and selected threshold.

    Saving the threshold in the checkpoint allows the Streamlit dashboard
    to use the same classification rule as the evaluation pipeline.
    """

    model_path = MODEL_DIR / "LSTM_PyTorch.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": 16,
            "prediction_threshold": float(prediction_threshold),
            "target_column": TARGET_COLUMN,
            "best_epoch": int(best_epoch),
            "random_state": RANDOM_STATE,
        },
        model_path,
    )

    print(f"\nSaved Model:\n{model_path}")

    scaler_path = MODEL_DIR / "LSTM_scaler.joblib"

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(f"\nSaved Scaler:\n{scaler_path}")


# =========================================================
# Main
# =========================================================

def main() -> None:
    """Train, validate, evaluate and save the PyTorch LSTM model."""

    print("=" * 60)
    print("PYTORCH LSTM INCIDENT PREDICTION")
    print("=" * 60)

    ensure_output_directories()

    device = get_device()

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Training device: {device}")

    (
        X_training_full,
        X_validation,
        X_test,
        y_training_full,
        y_validation,
        y_test,
        scaler,
    ) = prepare_lstm_data()

    print("\nChronological LSTM Dataset")
    print(f"X_training_full: {X_training_full.shape}")
    print(f"X_validation   : {X_validation.shape}")
    print(f"X_test         : {X_test.shape}")

    print("\nNatural Training Target Distribution")
    print(
        pd.Series(y_training_full.astype(int))
        .value_counts()
        .sort_index()
    )

    print("\nNatural Validation Target Distribution")
    print(
        pd.Series(y_validation.astype(int))
        .value_counts()
        .sort_index()
    )

    print("\nNatural Test Target Distribution")
    print(
        pd.Series(y_test.astype(int))
        .value_counts()
        .sort_index()
    )

    # Undersample only the actual training subset.
    X_training, y_training = undersample_training_data(
        X_training_full,
        y_training_full,
        NEGATIVE_TO_POSITIVE_RATIO,
    )

    print("\nUndersampled Training Dataset")
    print(f"X_training: {X_training.shape}")

    print(
        pd.Series(y_training.astype(int))
        .value_counts()
        .sort_index()
    )

    training_dataset = TensorDataset(
        torch.tensor(
            X_training,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_training,
            dtype=torch.float32,
        ),
    )

    validation_dataset = TensorDataset(
        torch.tensor(
            X_validation,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_validation,
            dtype=torch.float32,
        ),
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_test,
            dtype=torch.float32,
        ),
    )

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=data_loader_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = IncidentLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=16,
    ).to(device)

    print("\nModel Architecture")
    print(model)

    (
        training_losses,
        validation_losses,
        training_time,
        best_epoch,
    ) = train_lstm(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )

    save_training_history(
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
    )

    validation_probabilities = predict_probabilities(
        model=model,
        data_loader=validation_loader,
        device=device,
    )

    best_threshold = select_best_threshold(
        y_validation=y_validation,
        validation_probabilities=validation_probabilities,
    )

    save_model_checkpoint(
        model=model,
        scaler=scaler,
        prediction_threshold=best_threshold,
        best_epoch=best_epoch,
    )

    evaluate_lstm(
        model=model,
        test_loader=test_loader,
        y_test=y_test,
        device=device,
        training_time=training_time,
        prediction_threshold=best_threshold,
        best_epoch=best_epoch,
    )

    print("\nPyTorch LSTM completed successfully.")


if __name__ == "__main__":
    main()