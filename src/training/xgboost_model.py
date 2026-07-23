from xgboost import XGBClassifier

from src.utils.preprocessing import prepare_train_test
from src.training.train import train_model

print("=" * 60)
print("XGBOOST")
print("=" * 60)

X_train, X_test, y_train, y_test = prepare_train_test()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

train_model(
    model=model,
    model_name="XGBoost",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)