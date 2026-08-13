"""OOF stacking ensemble for the Titanic project."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PROCESSED_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
)
from src.modeling import build_preprocessor, load_modeling_data, WCGSurvivalEncoder, AgeImputer

LOGGER = logging.getLogger("titanic.stacking")
FEATURE_EXCLUSIONS = {TARGET_COLUMN, "PassengerId"}


def _base_models() -> Dict[str, Any]:
    """Build base estimators with safe CPU defaults."""
    models: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            alpha=0.1, batch_size=32, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            random_state=RANDOM_STATE, max_iter=500,
        ),
    }
    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=4,
            l2_leaf_reg=50.0, subsample=0.7,
            random_seed=RANDOM_STATE, verbose=False, task_type="CPU",
        )
    except ImportError:
        LOGGER.warning("CatBoost is unavailable; skipping it")
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=4, reg_lambda=50.0,
            subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, device="cpu", verbosity=-1,
        )
    except ImportError:
        LOGGER.warning("LightGBM is unavailable; skipping it")

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            reg_lambda=50.0, subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, tree_method="hist", eval_metric="logloss",
        )
    except ImportError:
        LOGGER.warning("XGBoost is unavailable; skipping it")

    return models


def _pipeline(model: Any, frame: pd.DataFrame) -> Pipeline:
    return ImbPipeline([
        ("wcg_encoder", WCGSurvivalEncoder()),
        ("age_imputer", AgeImputer(random_state=RANDOM_STATE)),
        ("preprocessor", build_preprocessor(frame)),
        ("model", model)
    ])


def _metrics(y_true: pd.Series, probabilities: np.ndarray) -> Dict[str, float]:
    labels = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1_macro": float(f1_score(y_true, labels, average="macro")),
    }


def generate_oof_predictions(
    X: pd.DataFrame, y: pd.Series, models: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, np.ndarray]]:
    """Generate one probability per row from each base model using OOF folds."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_STATE)
    oof = pd.DataFrame(index=X.index, columns=models.keys(), dtype=float)
    fold_scores: Dict[str, list[Dict[str, float]]] = {name: [] for name in models}
    for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        for name, estimator in models.items():
            fitted = _pipeline(clone(estimator), X.iloc[fit_idx])
            fitted.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            probabilities = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
            oof.iloc[valid_idx, oof.columns.get_loc(name)] = probabilities
            fold_scores[name].append(_metrics(y.iloc[valid_idx], probabilities))
        LOGGER.info("Generated OOF predictions for fold %d/5", fold)

    full_models: Dict[str, Pipeline] = {}
    for name, estimator in models.items():
        fitted = _pipeline(clone(estimator), X)
        fitted.fit(X, y)
        full_models[name] = fitted
        LOGGER.info(
            "%s OOF metrics: %s",
            name,
            {key: round(float(np.mean([score[key] for score in values])), 4)
             for key in ["accuracy", "roc_auc", "f1_macro"]
             for values in [fold_scores[name]]},
        )
    return oof, full_models, fold_scores


def run_stacking_pipeline() -> pd.DataFrame:
    """Train the OOF stacker and create the stacking submission."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    train, test = load_modeling_data()
    features = [column for column in train.columns if column not in FEATURE_EXCLUSIONS]
    X_train, y_train = train[features], train[TARGET_COLUMN].astype(int)
    X_test = test[features]

    models = _base_models()
    oof, full_models, fold_scores = generate_oof_predictions(X_train, y_train, models)
    if oof.isna().any().any():
        raise RuntimeError("OOF prediction matrix contains missing values")
    oof_path = Path(EXPERIMENTS_DIR) / "oof_predictions.npz"
    np.savez(oof_path, predictions=oof.to_numpy(), target=y_train.to_numpy())

    from scipy.optimize import minimize
    from sklearn.metrics import log_loss

    # We specifically optimize on XGBoost, LightGBM, and CatBoost
    blend_models = ["XGBoost", "LightGBM", "CatBoost"]
    blend_models = [m for m in blend_models if m in oof.columns]

    if not blend_models:
        LOGGER.warning("None of the target blend models found; falling back to uniform weights.")
        optimal_weights = np.ones(oof.shape[1]) / oof.shape[1]
        blend_oof = oof
    else:
        blend_oof = oof[blend_models]
        LOGGER.info("OOF Check - NaN count: %s, Data types: %s, Min: %s, Max: %s", blend_oof.isna().sum().to_dict(), blend_oof.dtypes.to_dict(), blend_oof.min().to_dict(), blend_oof.max().to_dict())
        def loss_func(weights):
            blended = np.dot(blend_oof, weights)
            return log_loss(y_train, blended)

        initial_weights = np.ones(len(blend_models)) / len(blend_models)
        bounds = [(0, 1) for _ in range(len(blend_models))]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        optimal_weights = res.x
        LOGGER.info("Optimized Blend Weights: %s", dict(zip(blend_models, optimal_weights)))

    meta_probabilities = np.dot(blend_oof, optimal_weights)
    stack_metrics = _metrics(y_train, meta_probabilities)
    stack_metrics["training_metrics"] = _metrics(y_train, meta_probabilities) # Optimization ran on full OOF directly
    LOGGER.info("SLSQP Blend OOF metrics: %s", stack_metrics)

    test_base = pd.DataFrame({
        name: full_models[name].predict_proba(X_test)[:, 1] for name in blend_models
    }) if blend_models else pd.DataFrame({
        name: model.predict_proba(X_test)[:, 1] for name, model in full_models.items()
    })

    test_predictions = np.dot(test_base, optimal_weights)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        TARGET_COLUMN: (test_predictions >= 0.5).astype(int),
    })

    submission_path = Path(SUBMISSIONS_DIR) / "submission_stacking.csv"
    submission.to_csv(submission_path, index=False)

    submission_prob = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        TARGET_COLUMN: test_predictions,
    })
    submission_prob_path = Path(SUBMISSIONS_DIR) / "submission_blend_probabilities.csv"
    submission_prob.to_csv(submission_prob_path, index=False)

    results: Dict[str, Any] = {
        "base_models": {
            name: {
                "accuracy": float(np.mean([score["accuracy"] for score in scores])),
                "roc_auc": float(np.mean([score["roc_auc"] for score in scores])),
                "f1_macro": float(np.mean([score["f1_macro"] for score in scores])),
            }
            for name, scores in fold_scores.items()
        },
        "stacker": stack_metrics,
        "submission": {"path": str(submission_path), "rows": len(submission)},
    }
    (Path(EXPERIMENTS_DIR) / "stacking_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    for name, model in full_models.items():
        joblib.dump(model, Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib")
    joblib.dump(optimal_weights, Path(MODELS_DIR) / "stacking_meta_model.joblib")
    LOGGER.info("Stacking submission saved to %s", submission_path)
    return submission


if __name__ == "__main__":
    run_stacking_pipeline()
