from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# LSTM Metadata
# PyTorch is lazy-loaded only when LSTM is selected.
# =========================================================

LSTM_SEQUENCE_LENGTH = 10

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


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_observability_metrics.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"

LOGISTIC_MODEL_FILE = (
    MODEL_DIRECTORY
    / "LogisticRegression.joblib"
)

RANDOM_FOREST_MODEL_FILE = (
    MODEL_DIRECTORY
    / "RandomForest.joblib"
)

XGBOOST_MODEL_FILE = (
    MODEL_DIRECTORY
    / "XGBoost.joblib"
)

LSTM_MODEL_FILE = (
    MODEL_DIRECTORY
    / "LSTM_PyTorch.pt"
)

LSTM_SCALER_FILE = (
    MODEL_DIRECTORY
    / "LSTM_scaler.joblib"
)

LSTM_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "LSTM.csv"
)

XGBOOST_PREDICTOR_FILE = (
    PROJECT_ROOT
    / "dashboard"
    / "xgboost_predictor.py"
)

MODEL_FILES = {
    "Logistic Regression": LOGISTIC_MODEL_FILE,
    "Random Forest": RANDOM_FOREST_MODEL_FILE,
    "XGBoost": XGBOOST_MODEL_FILE,
    "LSTM": LSTM_MODEL_FILE,
}


# =========================================================
# Streamlit Configuration
# =========================================================

st.set_page_config(
    page_title="AIOps Incident Prediction",
    page_icon="⚠️",
    layout="wide",
)


# =========================================================
# Data and Model Loading
# =========================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    dataframe = pd.read_csv(
        DATA_FILE
    )

    dataframe["Timestamp"] = pd.to_datetime(
        dataframe["Timestamp"],
        errors="raise",
    )

    return dataframe


@st.cache_resource
def load_classical_model(
    model_path: Path,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


@st.cache_resource
def load_lstm_model():
    """
    Import PyTorch only when LSTM is selected.
    """

    from lstm_utils import (
        load_lstm_resources,
    )

    return load_lstm_resources(
        model_path=LSTM_MODEL_FILE,
        scaler_path=LSTM_SCALER_FILE,
        metrics_path=LSTM_METRICS_FILE,
    )


# =========================================================
# Feature Helpers
# =========================================================

def get_default_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    excluded_columns = {
        "Timestamp",
        "Service",
        "Incident",
        "Severity",
        "Target_5min",
        "Target_10min",
        "Target_15min",
    }

    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]


def get_model_feature_columns(
    model,
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return the exact input features used during model fitting.
    """

    if hasattr(
        model,
        "feature_names_in_",
    ):
        return list(
            model.feature_names_in_
        )

    if hasattr(
        model,
        "named_steps",
    ):
        for step in model.named_steps.values():
            if hasattr(
                step,
                "feature_names_in_",
            ):
                return list(
                    step.feature_names_in_
                )

    return get_default_feature_columns(
        dataframe
    )


def get_risk_level(
    probability: float,
) -> tuple[str, str]:
    if probability >= 0.80:
        return "Critical", "🔴"

    if probability >= 0.60:
        return "High", "🟠"

    if probability >= 0.40:
        return "Moderate", "🟡"

    return "Low", "🟢"


# =========================================================
# Prediction Helpers
# =========================================================

def predict_with_classical_model(
    model,
    selected_row: pd.Series,
    feature_columns: list[str],
) -> tuple[int, float]:
    model_input = pd.DataFrame(
        [
            selected_row[
                feature_columns
            ].to_dict()
        ],
        columns=feature_columns,
    )

    prediction = int(
        model.predict(
            model_input
        )[0]
    )

    probability = float(
        model.predict_proba(
            model_input
        )[0, 1]
    )

    return (
        prediction,
        probability,
    )


def predict_with_lstm(
    model,
    scaler,
    service_dataframe: pd.DataFrame,
    selected_index: int,
    threshold: float,
) -> tuple[int, float]:
    from lstm_utils import (
        predict_lstm_incident,
    )

    return predict_lstm_incident(
        model=model,
        scaler=scaler,
        service_dataframe=service_dataframe,
        selected_index=selected_index,
        threshold=threshold,
    )


def get_xgboost_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return the expected XGBoost input feature set.

    This must match the features used when XGBoost was trained.
    Severity is intentionally excluded.
    """

    return get_default_feature_columns(
        dataframe
    )


def predict_with_xgboost_subprocess(
    selected_row: pd.Series,
    feature_columns: list[str],
) -> tuple[int, float]:
    """
    Execute XGBoost inference in a separate Python process.

    This isolates XGBoost's native OpenMP runtime from Streamlit
    and PyTorch, preventing process-level segmentation faults.
    """

    if not XGBOOST_PREDICTOR_FILE.exists():
        raise FileNotFoundError(
            "XGBoost predictor script not found: "
            f"{XGBOOST_PREDICTOR_FILE}"
        )

    feature_values = {
        feature: float(
            selected_row[feature]
        )
        for feature in feature_columns
    }

    completed_process = subprocess.run(
        [
            sys.executable,
            str(
                XGBOOST_PREDICTOR_FILE
            ),
            json.dumps(
                feature_values
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=str(
            PROJECT_ROOT
        ),
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "XGBoost prediction subprocess failed.\n\n"
            f"{completed_process.stderr}"
        )

    output_lines = [
        line.strip()
        for line
        in completed_process.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError(
            "XGBoost subprocess returned no output."
        )

    try:
        result = json.loads(
            output_lines[-1]
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Unable to parse XGBoost subprocess output.\n\n"
            f"Output:\n{completed_process.stdout}"
        ) from error

    return (
        int(
            result["prediction"]
        ),
        float(
            result["probability"]
        ),
    )


# =========================================================
# Load Dataset
# =========================================================

try:
    df = load_data()

except Exception as error:
    st.error(
        str(error)
    )
    st.stop()


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header(
    "Prediction Configuration"
)

selected_model_name = (
    st.sidebar.selectbox(
        "Select prediction model",
        options=list(
            MODEL_FILES.keys()
        ),
        index=0,
    )
)

selected_service = (
    st.sidebar.selectbox(
        "Select microservice",
        options=sorted(
            df["Service"].unique()
        ),
    )
)

view_mode = st.sidebar.radio(
    "Select dashboard mode",
    options=[
        "Incident Replay",
        "Full Timeline",
    ],
)


# =========================================================
# Load Selected Model
# =========================================================

try:
    if selected_model_name == "LSTM":
        (
            model,
            lstm_scaler,
            lstm_threshold,
        ) = load_lstm_model()

        feature_columns = list(
            LSTM_FEATURE_COLUMNS
        )

    elif selected_model_name == "XGBoost":
        # XGBoost is intentionally not loaded here.
        # It is invoked in a separate process.
        model = None

        feature_columns = (
            get_xgboost_feature_columns(
                df
            )
        )

    else:
        model = load_classical_model(
            MODEL_FILES[
                selected_model_name
            ]
        )

        feature_columns = (
            get_model_feature_columns(
                model=model,
                dataframe=df,
            )
        )

except Exception as error:
    st.error(
        f"Model loading failed: {error}"
    )
    st.stop()


# =========================================================
# Validate Required Features
# =========================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]

if missing_features:
    st.error(
        "The processed dataset is missing "
        "features required by the selected model: "
        f"{missing_features}"
    )
    st.stop()


# =========================================================
# Filter Selected Service
# =========================================================

service_df = (
    df[
        df["Service"]
        == selected_service
    ]
    .sort_values(
        "Timestamp"
    )
    .reset_index(
        drop=True
    )
)

if service_df.empty:
    st.error(
        f"No records found for "
        f"{selected_service}"
    )
    st.stop()


incident_indices = (
    service_df.index[
        service_df[
            "Incident"
        ] == 1
    ].tolist()
)


# =========================================================
# Select Observation
# =========================================================

if view_mode == "Incident Replay":

    if not incident_indices:
        st.warning(
            "No incidents were found for this service. "
            "Using timeline navigation."
        )

        minimum_index = (
            LSTM_SEQUENCE_LENGTH - 1
            if selected_model_name == "LSTM"
            else 0
        )

        selected_index = (
            st.sidebar.slider(
                "Select timeline position",
                min_value=minimum_index,
                max_value=len(
                    service_df
                ) - 1,
                value=max(
                    minimum_index,
                    min(
                        1000,
                        len(
                            service_df
                        ) - 1,
                    ),
                ),
                step=1,
            )
        )

    else:
        incident_number = (
            st.sidebar.selectbox(
                "Select incident event",
                options=list(
                    range(
                        1,
                        len(
                            incident_indices
                        ) + 1,
                    )
                ),
            )
        )

        incident_index = (
            incident_indices[
                incident_number - 1
            ]
        )

        minutes_before_incident = (
            st.sidebar.slider(
                "Minutes before incident",
                min_value=0,
                max_value=20,
                value=5,
                step=1,
            )
        )

        selected_index = max(
            0,
            incident_index
            - minutes_before_incident,
        )

        if (
            selected_model_name == "LSTM"
            and selected_index
            < LSTM_SEQUENCE_LENGTH - 1
        ):
            selected_index = (
                LSTM_SEQUENCE_LENGTH - 1
            )

else:
    minimum_index = (
        LSTM_SEQUENCE_LENGTH - 1
        if selected_model_name == "LSTM"
        else 0
    )

    selected_index = (
        st.sidebar.slider(
            "Select timeline position",
            min_value=minimum_index,
            max_value=len(
                service_df
            ) - 1,
            value=max(
                minimum_index,
                min(
                    1000,
                    len(
                        service_df
                    ) - 1,
                ),
            ),
            step=1,
        )
    )


selected_row = service_df.iloc[
    selected_index
]


# =========================================================
# Generate Prediction
# =========================================================

try:
    if selected_model_name == "LSTM":
        (
            prediction,
            incident_probability,
        ) = predict_with_lstm(
            model=model,
            scaler=lstm_scaler,
            service_dataframe=service_df,
            selected_index=selected_index,
            threshold=lstm_threshold,
        )

    elif selected_model_name == "XGBoost":
        (
            prediction,
            incident_probability,
        ) = predict_with_xgboost_subprocess(
            selected_row=selected_row,
            feature_columns=feature_columns,
        )

    else:
        (
            prediction,
            incident_probability,
        ) = predict_with_classical_model(
            model=model,
            selected_row=selected_row,
            feature_columns=feature_columns,
        )

except Exception as error:
    st.error(
        "Prediction failed.\n\n"
        f"{error}"
    )
    st.stop()


risk_level, risk_icon = (
    get_risk_level(
        incident_probability
    )
)


# =========================================================
# Header
# =========================================================

st.title(
    "Machine Learning-Based Incident Prediction"
)

st.caption(
    "A proof-of-concept AIOps dashboard for "
    "predicting incidents in distributed systems "
    "using observability data."
)

st.divider()


# =========================================================
# Context
# =========================================================

context_1, context_2, context_3 = (
    st.columns(3)
)

context_1.metric(
    "Selected Service",
    selected_service,
)

context_2.metric(
    "Timestamp",
    selected_row[
        "Timestamp"
    ].strftime(
        "%Y-%m-%d %H:%M"
    ),
)

context_3.metric(
    "Prediction Model",
    selected_model_name,
)


# =========================================================
# Current Observability Metrics
# =========================================================

st.subheader(
    "Current Observability Metrics"
)

metric_columns = st.columns(5)

metric_columns[0].metric(
    "CPU Utilization",
    f"{selected_row['CPU']:.1f}%",
)

metric_columns[1].metric(
    "Memory Utilization",
    f"{selected_row['Memory']:.1f}%",
)

metric_columns[2].metric(
    "Latency",
    f"{selected_row['Latency']:.1f} ms",
)

metric_columns[3].metric(
    "Throughput",
    (
        f"{selected_row['Throughput']:.1f} "
        "req/s"
    ),
)

metric_columns[4].metric(
    "Error Rate",
    f"{selected_row['ErrorRate']:.2f}%",
)


# =========================================================
# Prediction Result
# =========================================================

st.subheader(
    "Incident Prediction"
)

prediction_col, probability_col, risk_col = (
    st.columns(3)
)

prediction_col.metric(
    "Prediction",
    (
        "Incident Likely"
        if prediction == 1
        else "Normal Operation"
    ),
)

probability_col.metric(
    "Incident Probability",
    (
        f"{incident_probability * 100:.2f}%"
    ),
)

risk_col.metric(
    "Operational Risk",
    f"{risk_icon} {risk_level}",
)


if prediction == 1:
    st.error(
        "Early warning: the selected model "
        "predicts that an incident may occur "
        "within the next five minutes."
    )

else:
    st.success(
        "No immediate incident is predicted "
        "within the five-minute prediction horizon."
    )


if selected_model_name == "LSTM":
    st.info(
        "The LSTM uses the previous "
        f"{LSTM_SEQUENCE_LENGTH} minutes of "
        "observability data."
    )


if selected_model_name == "XGBoost":
    st.info(
        "XGBoost inference is executed in an isolated "
        "subprocess to avoid native OpenMP runtime conflicts "
        "with the Streamlit and PyTorch process."
    )


# =========================================================
# Ground Truth
# =========================================================

with st.expander(
    "Show experimental ground truth"
):
    ground_truth = {
        "Current incident marker": int(
            selected_row[
                "Incident"
            ]
        ),
        "Target within 5 minutes": int(
            selected_row[
                "Target_5min"
            ]
        ),
        "Target within 10 minutes": int(
            selected_row[
                "Target_10min"
            ]
        ),
        "Target within 15 minutes": int(
            selected_row[
                "Target_15min"
            ]
        ),
    }

    if "Severity" in selected_row.index:
        ground_truth[
            "Incident severity"
        ] = int(
            selected_row[
                "Severity"
            ]
        )

    st.write(
        ground_truth
    )

    actual_target = int(
        selected_row[
            "Target_5min"
        ]
    )

    if actual_target == prediction:
        st.success(
            "The prediction matches the "
            "five-minute ground-truth label."
        )

    else:
        st.warning(
            "The prediction does not match the "
            "five-minute ground-truth label."
        )


# =========================================================
# Metric Timeline
# =========================================================

st.subheader(
    "Recent Metric Behaviour"
)

history_start = max(
    0,
    selected_index - 30,
)

history_end = min(
    len(
        service_df
    ),
    selected_index + 6,
)

history_df = (
    service_df.iloc[
        history_start:
        history_end
    ]
    .copy()
    .set_index(
        "Timestamp"
    )
)

tab_1, tab_2, tab_3 = st.tabs(
    [
        "Resource Utilization",
        "Latency and Error Rate",
        "Throughput",
    ]
)

with tab_1:
    st.line_chart(
        history_df[
            [
                "CPU",
                "Memory",
            ]
        ]
    )

with tab_2:
    st.line_chart(
        history_df[
            [
                "Latency",
                "ErrorRate",
            ]
        ]
    )

with tab_3:
    st.line_chart(
        history_df[
            [
                "Throughput",
            ]
        ]
    )


# =========================================================
# Model Input Data
# =========================================================

with st.expander(
    "Show model input features"
):
    if selected_model_name == "LSTM":
        sequence_start = (
            selected_index
            - LSTM_SEQUENCE_LENGTH
            + 1
        )

        sequence_display = (
            service_df.iloc[
                sequence_start:
                selected_index + 1
            ][
                [
                    "Timestamp",
                    *LSTM_FEATURE_COLUMNS,
                ]
            ]
            .copy()
        )

        st.write(
            "Ten-minute sequence used "
            "by the LSTM:"
        )

        st.dataframe(
            sequence_display,
            width="stretch",
            hide_index=True,
        )

    else:
        feature_table = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Value": [
                    selected_row[
                        feature
                    ]
                    for feature
                    in feature_columns
                ],
            }
        )

        st.dataframe(
            feature_table,
            width="stretch",
            hide_index=True,
        )


# =========================================================
# Model Information
# =========================================================

with st.expander(
    "Show model information"
):
    model_information = {
        "Model": selected_model_name,
        "Prediction window": "5 minutes",
        "Dashboard mode": view_mode,
        "Number of input features": len(
            feature_columns
        ),
    }

    if selected_model_name == "LSTM":
        model_information.update(
            {
                "Framework": "PyTorch",
                "Sequence length": (
                    f"{LSTM_SEQUENCE_LENGTH} minutes"
                ),
                "Validation threshold": round(
                    lstm_threshold,
                    6,
                ),
            }
        )

    elif selected_model_name == "XGBoost":
        model_information.update(
            {
                "Framework": "XGBoost",
                "Serving mode": (
                    "Isolated Python subprocess"
                ),
            }
        )

    else:
        model_information[
            "Framework"
        ] = "Scikit-learn"

    st.write(
        model_information
    )

    st.write(
        "Input features:"
    )

    st.code(
        "\n".join(
            feature_columns
        )
    )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Dissertation POC developed by "
    "Koel Banerjee (2024AB05139), "
    "M.Tech Artificial Intelligence and "
    "Machine Learning, BITS Pilani."
)