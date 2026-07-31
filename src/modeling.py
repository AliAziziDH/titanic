"""Model comparison, validation, and submission generation for Titanic."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
    get_input_dir,
    LIGHTGBM_PARAMS,
    MODELS_DIR,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
)
from src.features import save_engineered_data

LOGGER = logging.getLogger("titanic.modeling")
MODEL_DIR = Path(MODELS_DIR)
CV_RESULTS_PATH = Path(EXPERIMENTS_DIR) / "cv_results.json"

NUMERICAL_FEATURES = [
    "Age", "Fare", "SibSp", "Parch", "Family_Size", "Ticket_Count", "Fare_per_Person",
]
CATEGORICAL_FEATURES = [
    "Sex", "Embarked", "Title_Num", "Title_Encoded", "Deck_Num", "Deck_Encoded",
    "Deck", "Deck_Group", "Family_Name", "Family_Size_Category", "Ticket_Prefix",
    "Fare_Bin", "Age_Band", "Sex_Pclass", "Title_Sex",
]
BINARY_FEATURES = ["Has_Cabin", "Is_Alone", "Is_Group", "Is_Mother"]


def load_modeling_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load engineered data so every fold imputes from its own training rows."""
    train_path = Path(DATA_PROCESSED_DIR) / "train_engineered.csv"
    test_path = Path(DATA_PROCESSED_DIR) / "test_engineered.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    input_dir = get_input_dir()
    raw_train = pd.read_csv(input_dir / "train.csv")
    raw_test = pd.read_csv(input_dir / "test.csv")
    train_engineered_path, test_engineered_path = save_engineered_data(
        raw_train, raw_test
    )
    return pd.read_csv(train_engineered_path), pd.read_csv(test_engineered_path)


def _available_columns(frame: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    numeric = [column for column in NUMERICAL_FEATURES if column in frame]
    categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
    binary = [column for column in BINARY_FEATURES if column in frame]
    return numeric, categorical, binary


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted independently within each CV fold."""
    numeric, categorical, binary = _available_columns(frame)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical + binary),
        ],
        remainder="drop",
    )


def _optional_models() -> Dict[str, Any]:
    """Return available optional gradient-boosting estimators with safe CPU defaults."""
    models: Dict[str, Any] = {}
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=RANDOM_STATE, verbose=False, task_type="CPU",
        )
    except ImportError:
        LOGGER.warning("CatBoost is unavailable; skipping it")
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_STATE, tree_method="hist", eval_metric="logloss",
        )
    except ImportError:
        LOGGER.warning("XGBoost is unavailable; skipping it")
    try:
        from lightgbm import LGBMClassifier

        params = dict(LIGHTGBM_PARAMS)
        params.update({"device": "cpu", "verbosity": -1})
        models["LightGBM"] = LGBMClassifier(**params)
    except ImportError:
        LOGGER.warning("LightGBM is unavailable; skipping it")
    return models


def default_models() -> Dict[str, Any]:
    """Return the configured model candidates."""
    models = _optional_models()
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return models


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: RepeatedStratifiedKFold,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate a model with preprocessing fitted separately on each fold."""
    scores = {"accuracy": [], "roc_auc": [], "f1_macro": []}
    for fold, (fit_idx, validation_idx) in enumerate(cv_strategy.split(X_train, y_train), start=1):
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(model)),
        ])
        pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        predictions = pipeline.predict(X_train.iloc[validation_idx])
        probabilities = pipeline.predict_proba(X_train.iloc[validation_idx])[:, 1]
        scores["accuracy"].append(accuracy_score(y_train.iloc[validation_idx], predictions))
        scores["roc_auc"].append(roc_auc_score(y_train.iloc[validation_idx], probabilities))
        scores["f1_macro"].append(f1_score(y_train.iloc[validation_idx], predictions, average="macro"))
        LOGGER.debug("%s fold %d complete", model_name, fold)
    result = {
        "model": model_name,
        "metrics": {
            metric: {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
            for metric, values in scores.items()
        },
    }
    LOGGER.info(
        "%s: accuracy=%.4f (+/- %.4f), roc_auc=%.4f (+/- %.4f), f1_macro=%.4f (+/- %.4f)",
        model_name,
        result["metrics"]["accuracy"]["mean"], result["metrics"]["accuracy"]["std"],
        result["metrics"]["roc_auc"]["mean"], result["metrics"]["roc_auc"]["std"],
        result["metrics"]["f1_macro"]["mean"], result["metrics"]["f1_macro"]["std"],
    )
    return result


def _select_best(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Select by ROC-AUC, with accuracy and macro-F1 as tie-breakers."""
    return max(
        results,
        key=lambda result: (
            result["metrics"]["roc_auc"]["mean"],
            result["metrics"]["accuracy"]["mean"],
            result["metrics"]["f1_macro"]["mean"],
        ),
    )


def run_modeling_pipeline() -> pd.DataFrame:
    """Train candidate models, select the best, and write a Kaggle submission."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    train, test = load_modeling_data()
    if TARGET_COLUMN not in train:
        raise ValueError(f"{TARGET_COLUMN} is missing from training data")
    feature_columns = [column for column in train.columns if column not in {TARGET_COLUMN, "PassengerId"}]
    X_train, y_train = train[feature_columns], train[TARGET_COLUMN].astype(int)
    X_test = test[feature_columns]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    results = [
        evaluate_model(model, X_train, y_train, cv, name)
        for name, model in default_models().items()
    ]
    CV_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = _select_best(results)
    best_model = default_models()[best["model"]]
    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", best_model),
    ])
    final_pipeline.fit(X_train, y_train)
    test_predictions = final_pipeline.predict(X_test).astype(int)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], TARGET_COLUMN: test_predictions})
    submission_path = Path(SUBMISSIONS_DIR) / "submission_modeling.csv"
    submission.to_csv(submission_path, index=False)
    joblib.dump(final_pipeline, MODEL_DIR / f"{best['model'].lower()}_final.joblib")
    (MODEL_DIR / "best_model.json").write_text(
        json.dumps({"model": best["model"], "metrics": best["metrics"]}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Selected %s; submission saved to %s", best["model"], submission_path)
    return submission


if __name__ == "__main__":
    run_modeling_pipeline()
