from sklearn.ensemble import RandomForestClassifier

from src.utils.preprocessing import prepare_train_test
from src.training.train import train_model

print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

X_train, X_test, y_train, y_test = prepare_train_test()

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=12,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1

)

train_model(

    model=model,

    model_name="RandomForest",

    X_train=X_train,

    X_test=X_test,

    y_train=y_train,

    y_test=y_test

)