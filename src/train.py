#Saving the tuned model pipeline

import mlflow
import mlflow.sklearn
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS,
    na_values=["?"],
    skipinitialspace=True,
)

for col in COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["num"]).copy()
df["target"] = (df["num"] > 0).astype(int)

X = df.drop(columns=["num", "target"])
y = df["target"]

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]
categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Tune Logistic Regression
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42,
    )),
])

lr_search = GridSearchCV(
    lr_pipeline,
    param_grid={
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__class_weight": [None, "balanced"],
    },
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)

lr_search.fit(X_train, y_train)

# Tune Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])

rf_search = GridSearchCV(
    rf_pipeline,
    param_grid={
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    },
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)

rf_search.fit(X_train, y_train)

searches = {
    "Logistic Regression": lr_search,
    "Random Forest": rf_search,
}

# Choose based on CV ROC-AUC; prefer Logistic Regression in a tie.
best_name = max(
    searches,
    key=lambda name: (
        searches[name].best_score_,
        name == "Logistic Regression",
    ),
)

best_search = searches[best_name]
best_model = best_search.best_estimator_

# Evaluate the selected model on the held-out test set once.
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "selected_model": best_name,
    "cv_roc_auc": float(best_search.best_score_),
    "test_accuracy": float(accuracy_score(y_test, predictions)),
    "test_precision": float(precision_score(
        y_test, predictions, zero_division=0
    )),
    "test_recall": float(recall_score(
        y_test, predictions, zero_division=0
    )),
    "test_f1": float(f1_score(
        y_test, predictions, zero_division=0
    )),
    "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
}

joblib.dump(best_model, ARTIFACT_DIR / "heart_disease_pipeline.joblib")

with open(ARTIFACT_DIR / "evaluation_metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)

with open(ARTIFACT_DIR / "model_metadata.json", "w") as file:
    json.dump({
        "selected_model": best_name,
        "best_parameters": best_search.best_params_,
        "input_features": list(X.columns),
        "target_definition": "0 = no disease; 1 = disease present",
        "dataset": "UCI Heart Disease - Cleveland",
        "random_state": 42,
    }, file, indent=2, default=str)

print("Selected model:", best_name)
print("Best CV ROC-AUC:", round(best_search.best_score_, 4))
print("Best parameters:", best_search.best_params_)
print("Test metrics:")
print(json.dumps(metrics, indent=2))
print("Saved model:", ARTIFACT_DIR / "heart_disease_pipeline.joblib")

# --------------------------------------------------
# MLflow experiment tracking
# --------------------------------------------------

# Store MLflow's database in the project root.
tracking_db = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(
    f"sqlite:///{tracking_db.as_posix()}"
)

# Group related runs under one experiment.
mlflow.set_experiment("Heart-Disease-Classification")

with mlflow.start_run(run_name=best_name.replace(" ", "_")):

    # Record the model and training configuration.
    mlflow.log_param("selected_model", best_name)
    mlflow.log_param("dataset", "UCI Heart Disease - Cleveland")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("cv_folds", 5)

    # Record the best hyperparameters.
    for parameter, value in best_search.best_params_.items():
        mlflow.log_param(parameter, str(value))

    # Record evaluation metrics.
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "selected_model"
    }
    mlflow.log_metrics(numeric_metrics)

    # Record useful context.
    mlflow.set_tag(
        "target_definition",
        "0 = no disease; 1 = disease present",
    )
    mlflow.set_tag("dataset_source", "UCI Heart Disease")

    # Save the trained sklearn pipeline to the MLflow run.
    mlflow.sklearn.log_model(
    sk_model=best_model,
    name="model",
    serialization_format="skops",
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree",
    ],
)

    # Attach the evaluation and metadata files to the run.
    mlflow.log_artifact(
        str(ARTIFACT_DIR / "evaluation_metrics.json")
    )
    mlflow.log_artifact(
        str(ARTIFACT_DIR / "model_metadata.json")
    )

    print("MLflow run ID:", mlflow.active_run().info.run_id)

print("MLflow tracking URI:", mlflow.get_tracking_uri())