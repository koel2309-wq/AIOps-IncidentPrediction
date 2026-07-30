from pathlib import Path

# -----------------------------------------------------
# Project Root
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("Current file:", Path(__file__).resolve())
print("PROJECT_ROOT:", PROJECT_ROOT)

# -----------------------------------------------------
# Data
# -----------------------------------------------------

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "observability_metrics.csv"

ENGINEERED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_observability_metrics.csv"
)

LABELED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labeled_observability_metrics.csv"
)

# -----------------------------------------------------
# Models
# -----------------------------------------------------

MODEL_DIR = PROJECT_ROOT / "models"

# -----------------------------------------------------
# Results
# -----------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_DIR = RESULTS_DIR / "metrics"

ROC_DIR = RESULTS_DIR / "roc_curves"

CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

FEATURE_IMPORTANCE_DIR = (
    RESULTS_DIR / "feature_importance"
)

# -----------------------------------------------------
# Create folders automatically
# -----------------------------------------------------

for directory in [

    MODEL_DIR,

    METRICS_DIR,

    ROC_DIR,

    CONFUSION_DIR,

    FEATURE_IMPORTANCE_DIR

]:
    directory.mkdir(parents=True, exist_ok=True)