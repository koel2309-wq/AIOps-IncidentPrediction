from __future__ import annotations

import random
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    LABELED_DATA,
    MODEL_DIR,
    METRICS_DIR,
    ROC_DIR,
    CONFUSION_DIR,
    RESULTS_DIR,
)


# =========================================================
# Configuration
# =========================================================

RANDOM_STATE = 42
TARGET_COLUMN = "Target_5min"

SEQUENCE_LENGTH = 10
TRAIN_RATIO = 0.80

EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Keep all positive sequences and five negative sequences
# for every positive sequence in the training set.
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
# Device Selection
# =========================================================

def get_device() -> torch.device:
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

    The target associated with each sequence is the label of the
    final timestamp in that sequence.
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
        (
            sequence_count,
            sequence_length,
            features.shape[1],
        ),
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
    Retain all positive sequences and randomly sample normal sequences.

    Only the training data is undersampled. Validation and test sets
    preserve their natural class distribution.
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
        [
            positive_indices,
            sampled_negative_indices,
        ]
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
    StandardScaler,
]:
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

    # Fill missing telemetry independently for every service.
    df[FEATURE_COLUMNS] = (
        df.groupby("Service")[FEATURE_COLUMNS]
        .transform(lambda group: group.ffill().bfill())
    )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    train_frames = []
    test_frames = []

    # Chronological 80/20 split per service.
    for _, service_df in df.groupby(
        "Service",
        sort=False,
    ):
        service_df = service_df.sort_values(
            "Timestamp"
        ).reset_index(drop=True)

        split_index = int(
            len(service_df) * TRAIN_RATIO
        )

        train_frames.append(
            service_df.iloc[:split_index].copy()
        )

        test_frames.append(
            service_df.iloc[split_index:].copy()
        )

    train_df = pd.concat(
        train_frames,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_frames,
        ignore_index=True,
    )

    scaler = StandardScaler()

    # Fit only on training data to avoid leakage.
    scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    X_train_parts = []
    y_train_parts = []

    X_test_parts = []
    y_test_parts = []

    for service in df["Service"].unique():
        service_train = train_df[
            train_df["Service"] == service
        ]

        service_test = test_df[
            test_df["Service"] == service
        ]

        train_features = scaler.transform(
            service_train[FEATURE_COLUMNS]
        ).astype(np.float32)

        test_features = scaler.transform(
            service_test[FEATURE_COLUMNS]
        ).astype(np.float32)

        train_targets = (
            service_train[TARGET_COLUMN]
            .astype(np.float32)
            .to_numpy()
        )

        test_targets = (
            service_test[TARGET_COLUMN]
            .astype(np.float32)
            .to_numpy()
        )

        X_service_train, y_service_train = create_sequences(
            train_features,
            train_targets,
            SEQUENCE_LENGTH,
        )

        X_service_test, y_service_test = create_sequences(
            test_features,
            test_targets,
            SEQUENCE_LENGTH,
        )

        X_train_parts.append(X_service_train)
        y_train_parts.append(y_service_train)

        X_test_parts.append(X_service_test)
        y_test_parts.append(y_service_test)

    return (
        np.concatenate(X_train_parts),
        np.concatenate(X_test_parts),
        np.concatenate(y_train_parts),
        np.concatenate(y_test_parts),
        scaler,
    )


# =========================================================
# PyTorch LSTM Model
# =========================================================

class IncidentLSTM(nn.Module):
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
) -> tuple[list[float], list[float], float]:
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    training_losses = []
    validation_losses = []

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

    training_time = (
        time.perf_counter()
        - training_start
    )

    return (
        training_losses,
        validation_losses,
        training_time,
    )


# =========================================================
# Probability Prediction
# =========================================================

def predict_probabilities(
    model: IncidentLSTM,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    probability_batches = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probabilities = torch.sigmoid(
                logits
            )

            probability_batches.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(
        probability_batches
    )


# =========================================================
# Validation-Based Threshold Selection
# =========================================================

def select_best_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
) -> float:
    """
    Select the probability threshold that maximizes validation F1-score.
    """

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_validation.astype(int),
            validation_probabilities,
        )
    )

    if len(thresholds) == 0:
        print(
            "No thresholds were generated. "
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
        np.argmax(f1_values)
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
) -> dict[str, float]:
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

    y_test_int = y_test.astype(int)

    # --------------------------------------------------
    # Save LSTM predictions for combined ROC comparison
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
            "Actual": y_test_int.reshape(-1),
            "Prediction": predictions.reshape(-1),
            "Probability": probabilities.reshape(-1),
        }
    )

    prediction_file = (
        predictions_dir
        / "LSTM_predictions.csv"
    )

    prediction_results.to_csv(
        prediction_file,
        index=False,
    )

    print(
        f"\nSaved Predictions:\n{prediction_file}"
    )

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

    roc_auc = roc_auc_score(
        y_test_int,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test_int,
        predictions,
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

    print(
        f"\nTraining Time : "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Inference Time: "
        f"{inference_time:.2f} seconds"
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Prediction_Threshold": prediction_threshold,
        "Training_Time_Seconds": training_time,
        "Inference_Time_Seconds": inference_time,
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / "LSTM.csv",
        index=False,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Normal",
            "Incident",
        ],
    )

    display.plot(
        values_format="d"
    )

    plt.title("PyTorch LSTM Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_DIR / "LSTM.png",
        dpi=300,
    )

    plt.close()

    RocCurveDisplay.from_predictions(
        y_test_int,
        probabilities,
        name="LSTM",
    )

    plt.title("PyTorch LSTM ROC Curve")
    plt.tight_layout()

    plt.savefig(
        ROC_DIR / "LSTM.png",
        dpi=300,
    )

    plt.close()

    return metrics


# =========================================================
# Training History
# =========================================================

def save_training_history(
    training_losses: list[float],
    validation_losses: list[float],
) -> None:
    history_directory = (
        RESULTS_DIR / "training_history"
    )

    history_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        training_losses,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        range(1, EPOCHS + 1),
        validation_losses,
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("PyTorch LSTM Training History")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        history_directory / "LSTM_loss.png",
        dpi=300,
    )

    plt.close()


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=" * 60)
    print("PYTORCH LSTM INCIDENT PREDICTION")
    print("=" * 60)

    device = get_device()

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Training device: {device}")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    ) = prepare_lstm_data()

    print("\nOriginal LSTM Dataset")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")

    print("\nOriginal Training Target Distribution")
    print(
        pd.Series(y_train.astype(int))
        .value_counts()
        .sort_index()
    )

    X_train, y_train = undersample_training_data(
        X_train,
        y_train,
        NEGATIVE_TO_POSITIVE_RATIO,
    )

    print("\nUndersampled Training Dataset")
    print(f"X_train: {X_train.shape}")

    print(
        pd.Series(y_train.astype(int))
        .value_counts()
        .sort_index()
    )

    # Validation split from the undersampled training set.
    validation_size = int(
        len(X_train) * 0.20
    )

    X_validation = X_train[:validation_size]
    y_validation = y_train[:validation_size]

    X_training = X_train[validation_size:]
    y_training = y_train[validation_size:]

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

    train_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    ) = train_lstm(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )

    model_path = (
        MODEL_DIR / "LSTM_PyTorch.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": 16,
        },
        model_path,
    )

    print(f"\nSaved model to:\n{model_path}")

    scaler_path = (
        MODEL_DIR / "LSTM_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(f"\nSaved scaler to:\n{scaler_path}")

    save_training_history(
        training_losses,
        validation_losses,
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

    evaluate_lstm(
        model=model,
        test_loader=test_loader,
        y_test=y_test,
        device=device,
        training_time=training_time,
        prediction_threshold=best_threshold,
    )

    print("\nPyTorch LSTM completed successfully.")


if __name__ == "__main__":
    main()