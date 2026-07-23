from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


# =========================================================
# Stable CPU-Only Inference for Streamlit on macOS
# =========================================================

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # This setting may already be initialized during a Streamlit rerun.
    pass

DEVICE = torch.device("cpu")


# =========================================================
# LSTM Configuration
# =========================================================

LSTM_FEATURE_COLUMNS = [
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

LSTM_SEQUENCE_LENGTH = 10


# =========================================================
# Model Definition
# =========================================================

class IncidentLSTM(nn.Module):
    """
    PyTorch LSTM model used for temporal incident prediction.
    """

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
# Resource Loading
# =========================================================

def load_lstm_resources(
    model_path: Path,
    scaler_path: Path,
    metrics_path: Path,
) -> tuple[IncidentLSTM, object, float]:
    """
    Load the trained LSTM model, StandardScaler, and the
    validation-selected classification threshold.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {model_path}"
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"LSTM scaler not found: {scaler_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    required_checkpoint_fields = {
        "model_state_dict",
        "input_size",
        "hidden_size",
    }

    missing_checkpoint_fields = (
        required_checkpoint_fields
        - set(checkpoint.keys())
    )

    if missing_checkpoint_fields:
        raise ValueError(
            "The LSTM checkpoint is missing required fields: "
            f"{sorted(missing_checkpoint_fields)}"
        )

    model = IncidentLSTM(
        input_size=int(
            checkpoint["input_size"]
        ),
        hidden_size=int(
            checkpoint["hidden_size"]
        ),
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    scaler = joblib.load(
        scaler_path
    )

    threshold = 0.5

    if metrics_path.exists():
        metrics_df = pd.read_csv(
            metrics_path
        )

        if (
            not metrics_df.empty
            and "Prediction_Threshold"
            in metrics_df.columns
        ):
            threshold_value = metrics_df.iloc[0][
                "Prediction_Threshold"
            ]

            if pd.notna(threshold_value):
                threshold = float(
                    threshold_value
                )

    return model, scaler, threshold


# =========================================================
# LSTM Prediction
# =========================================================

def predict_lstm_incident(
    model: IncidentLSTM,
    scaler,
    service_dataframe: pd.DataFrame,
    selected_index: int,
    threshold: float,
) -> tuple[int, float]:
    """
    Predict whether an incident is likely within the next
    five minutes using the previous ten observations.
    """

    minimum_index = (
        LSTM_SEQUENCE_LENGTH - 1
    )

    if selected_index < minimum_index:
        raise ValueError(
            "The LSTM requires at least "
            f"{LSTM_SEQUENCE_LENGTH} historical observations. "
            f"Select timeline index {minimum_index} or later."
        )

    sequence_start = (
        selected_index
        - LSTM_SEQUENCE_LENGTH
        + 1
    )

    missing_columns = [
        column
        for column in LSTM_FEATURE_COLUMNS
        if column not in service_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dashboard dataset is missing LSTM features: "
            f"{missing_columns}"
        )

    sequence_df = service_dataframe.iloc[
        sequence_start:selected_index + 1
    ][LSTM_FEATURE_COLUMNS].copy()

    if len(sequence_df) != LSTM_SEQUENCE_LENGTH:
        raise ValueError(
            "Unable to construct the required "
            f"{LSTM_SEQUENCE_LENGTH}-step LSTM sequence."
        )

    sequence_df = (
        sequence_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .ffill()
        .bfill()
        .fillna(0)
    )

    scaled_sequence = scaler.transform(
        sequence_df
    ).astype(
        np.float32,
        copy=False,
    )

    sequence_tensor = torch.from_numpy(
        scaled_sequence
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(
            sequence_tensor
        )

        probability = float(
            torch.sigmoid(logits)
            .cpu()
            .item()
        )

    prediction = int(
        probability >= threshold
    )

    return prediction, probability