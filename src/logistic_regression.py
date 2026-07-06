from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression

from preprocessing import prepare_train_test
from evaluation import evaluate_model

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

print("=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

X_train, X_test, y_train, y_test = prepare_train_test("Target_5min")

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Training Completed.")

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------

metrics = evaluate_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_name="LogisticRegression"
)

# ---------------------------------------------------
# Save Metrics
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

metrics_folder = BASE_DIR / "results" / "metrics"
metrics_folder.mkdir(parents=True, exist_ok=True)

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(
    metrics_folder / "logistic_regression_metrics.csv",
    index=False
)

print("\nMetrics saved successfully!")

print("=" * 60)