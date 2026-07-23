from __future__ import annotations

import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# Force TensorFlow to use CPU for stability on macOS.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

# Limit TensorFlow thread usage to avoid macOS thread stalls.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Disable GPU/Metal temporarily if TensorFlow detects it.
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("TensorFlow version:", tf.__version__)
print("TensorFlow devices:", tf.config.list_physical_devices())

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler

from src.config import (
    LABELED_DATA,
    MODEL_DIR,
    METRICS_DIR,
    ROC_DIR,
    CONFUSION_DIR,
    RESULTS_DIR,
)


# =========================================================
# Reproducibility
# =========================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================================================
# LSTM Configuration
# =========================================================

TARGET_COLUMN = "Target_5min"

# Previous 10 minutes are used to predict an upcoming incident.
SEQUENCE_LENGTH = 10

# Chronological train-test split for each service.
TRAIN_RATIO = 0.80

# Reduced values for faster laptop training.
EPOCHS = 3
BATCH_SIZE = 64

# Retain all positive samples and use 20 normal samples
# for each positive sample in the training set.
NEGATIVE_TO_POSITIVE_RATIO = 5

PREDICTION_THRESHOLD = 0.50


# Focused feature set for efficient temporal modeling.
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
# Sequence Creation
# =========================================================

def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert row-wise observations into fixed-length time sequences.

    Each sample contains `sequence_length` consecutive observations.
    The target corresponds to the final timestamp of the sequence.
    """

    if len(features) < sequence_length:
        return (
            np.empty(
                (0, sequence_length, features.shape[1]),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int32),
        )

    sequence_count = len(features) - sequence_length + 1

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
        dtype=np.int32,
    )

    for sequence_index, end_index in enumerate(
        range(sequence_length - 1, len(features))
    ):
        start_index = end_index - sequence_length + 1

        X_sequences[sequence_index] = features[
            start_index:end_index + 1
        ]

        y_sequences[sequence_index] = targets[end_index]

    return X_sequences, y_sequences


# =========================================================
# Training Undersampling
# =========================================================

def undersample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    negative_to_positive_ratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retain every positive sequence and randomly sample normal sequences.

    Only the training set is undersampled. The test set remains unchanged
    so that evaluation reflects the original class distribution.
    """

    positive_indices = np.where(y_train == 1)[0]
    negative_indices = np.where(y_train == 0)[0]

    if len(positive_indices) == 0:
        raise ValueError(
            "No positive incident samples were found in the training set."
        )

    maximum_negative_samples = (
        len(positive_indices)
        * negative_to_positive_ratio
    )

    negative_sample_count = min(
        maximum_negative_samples,
        len(negative_indices),
    )

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    selected_negative_indices = random_generator.choice(
        negative_indices,
        size=negative_sample_count,
        replace=False,
    )

    selected_indices = np.concatenate(
        [
            positive_indices,
            selected_negative_indices,
        ]
    )

    random_generator.shuffle(selected_indices)

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
    """
    Load the labeled dataset, perform a chronological split,
    scale the features, and create sequences independently
    for each microservice.
    """

    print(f"\nLoading labeled data from:\n{LABELED_DATA}")

    if not LABELED_DATA.exists():
        raise FileNotFoundError(
            f"Labeled dataset was not found: {LABELED_DATA}"
        )

    df = pd.read_csv(LABELED_DATA)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="raise",
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
            f"Missing required dataset columns: {missing_columns}"
        )

    df = df.sort_values(
        ["Service", "Timestamp"]
    ).reset_index(drop=True)

    # Handle missing telemetry independently for each service.
    df[FEATURE_COLUMNS] = (
        df.groupby("Service")[FEATURE_COLUMNS]
        .transform(lambda group: group.ffill().bfill())
    )

    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    # Chronological split per service prevents leakage
    # from future observations into training.
    for service, service_df in df.groupby(
        "Service",
        sort=False,
    ):
        service_df = service_df.sort_values(
            "Timestamp"
        ).reset_index(drop=True)

        split_index = int(
            len(service_df) * TRAIN_RATIO
        )

        if split_index < SEQUENCE_LENGTH:
            raise ValueError(
                f"Insufficient training observations for service: {service}"
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

    # Fit scaler only on training observations.
    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []

    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []

    for service in df["Service"].unique():
        service_train = train_df[
            train_df["Service"] == service
        ].copy()

        service_test = test_df[
            test_df["Service"] == service
        ].copy()

        train_features = scaler.transform(
            service_train[FEATURE_COLUMNS]
        ).astype(np.float32)

        test_features = scaler.transform(
            service_test[FEATURE_COLUMNS]
        ).astype(np.float32)

        train_targets = (
            service_train[TARGET_COLUMN]
            .astype(int)
            .to_numpy()
        )

        test_targets = (
            service_test[TARGET_COLUMN]
            .astype(int)
            .to_numpy()
        )

        X_service_train, y_service_train = (
            create_sequences(
                features=train_features,
                targets=train_targets,
                sequence_length=SEQUENCE_LENGTH,
            )
        )

        X_service_test, y_service_test = (
            create_sequences(
                features=test_features,
                targets=test_targets,
                sequence_length=SEQUENCE_LENGTH,
            )
        )

        X_train_parts.append(X_service_train)
        y_train_parts.append(y_service_train)

        X_test_parts.append(X_service_test)
        y_test_parts.append(y_service_test)

    X_train = np.concatenate(
        X_train_parts,
        axis=0,
    )

    y_train = np.concatenate(
        y_train_parts,
        axis=0,
    )

    X_test = np.concatenate(
        X_test_parts,
        axis=0,
    )

    y_test = np.concatenate(
        y_test_parts,
        axis=0,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    )


# =========================================================
# Model Definition
# =========================================================

def build_lstm_model(
    sequence_length: int,
    feature_count: int,
) -> tf.keras.Model:
    """
    Build a compact LSTM model suitable for CPU-based training.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    sequence_length,
                    feature_count,
                )
            ),

            tf.keras.layers.LSTM(
                units=16,
                return_sequences=False,
            ),

            tf.keras.layers.Dropout(
                rate=0.20
            ),

            tf.keras.layers.Dense(
                units=8,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="roc_auc"
            ),
        ],
        run_eagerly=True,
    )

    return model


# =========================================================
# Training History Figures
# =========================================================

def save_training_history(
    history: tf.keras.callbacks.History,
) -> None:
    """
    Save training and validation loss and ROC-AUC plots.
    """

    history_dir = (
        RESULTS_DIR / "training_history"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Loss figure
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        history_dir / "LSTM_loss.png",
        dpi=300,
    )

    plt.close()

    # ROC-AUC history figure
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["roc_auc"],
        label="Training ROC-AUC",
    )

    plt.plot(
        history.history["val_roc_auc"],
        label="Validation ROC-AUC",
    )

    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("LSTM Training and Validation ROC-AUC")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        history_dir / "LSTM_roc_auc_history.png",
        dpi=300,
    )

    plt.close()


# =========================================================
# Evaluation
# =========================================================

def evaluate_lstm(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float,
) -> dict[str, float]:
    """
    Evaluate the LSTM on the unchanged chronological test set.
    """

    prediction_start = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    inference_time = (
        time.perf_counter()
        - prediction_start
    )

    predictions = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    print("\n" + "=" * 50)
    print("LSTM")
    print("=" * 50)

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
        "Training_Time_Seconds": training_time,
        "Inference_Time_Seconds": inference_time,
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / "LSTM.csv",
        index=False,
    )

    # Confusion matrix
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

    plt.title("LSTM Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_DIR / "LSTM.png",
        dpi=300,
    )

    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name="LSTM",
    )

    plt.title("LSTM ROC Curve")
    plt.tight_layout()

    plt.savefig(
        ROC_DIR / "LSTM.png",
        dpi=300,
    )

    plt.close()

    return metrics


# =========================================================
# Main Training Process
# =========================================================

def main() -> None:
    print("=" * 60)
    print("LSTM TEMPORAL INCIDENT PREDICTION MODEL")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    ) = prepare_lstm_data()

    print("\nOriginal LSTM Dataset Shapes")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    print("\nOriginal Training Target Distribution")
    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    # Reduce only the training dataset.
    X_train, y_train = undersample_training_data(
        X_train=X_train,
        y_train=y_train,
        negative_to_positive_ratio=(
            NEGATIVE_TO_POSITIVE_RATIO
        ),
    )

    print("\nUndersampled Training Dataset")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print("\nUndersampled Target Distribution")
    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    print("\nFull Test Target Distribution")
    print(
        pd.Series(y_test)
        .value_counts()
        .sort_index()
    )

    model = build_lstm_model(
        sequence_length=X_train.shape[1],
        feature_count=X_train.shape[2],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
    ]

    training_start = time.perf_counter()

    print("\nTesting one training batch...")

    batch_result = model.train_on_batch(
        X_train[:64],
        y_train[:64],
    )

    print("One-batch result:", batch_result)
    print("Starting full LSTM training...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    model_path = (
        MODEL_DIR / "LSTM.keras"
    )

    model.save(model_path)

    print(f"\nSaved LSTM model to:\n{model_path}")

    scaler_path = (
        MODEL_DIR / "LSTM_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(f"\nSaved LSTM scaler to:\n{scaler_path}")

    save_training_history(history)

    evaluate_lstm(
        model=model,
        X_test=X_test,
        y_test=y_test,
        training_time=training_time,
    )

    print("\nLSTM training completed successfully.")


if __name__ == "__main__":
    main()