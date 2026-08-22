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
from src.modeling import build_preprocessor, load_modeling_data, WCGSurvivalEncoder, AgeImputer, compute_ipw_weights
from src.csa_allocator import ConfidentSinkhornAllocator

LOGGER = logging.getLogger("titanic.stacking")
FEATURE_EXCLUSIONS = {TARGET_COLUMN, "PassengerId"}


def _base_models() -> Dict[str, Any]:
    """Build base estimators with safe CPU defaults."""
    models: Dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1
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
            n_jobs=1,
        )
    except ImportError:
        LOGGER.warning("LightGBM is unavailable; skipping it")

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            reg_lambda=50.0, subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, tree_method="hist", eval_metric="logloss",
            n_jobs=1,
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

    from sklearn.linear_model import LogisticRegression

    # We specifically optimize on XGBoost, LightGBM, and CatBoost
    blend_models = ["XGBoost", "LightGBM", "CatBoost"]
    blend_models = [m for m in blend_models if m in oof.columns]

    if not blend_models:
        LOGGER.warning("None of the target blend models found; falling back to all models.")
        blend_oof = oof
        blend_models = list(oof.columns)
    else:
        blend_oof = oof[blend_models]
        LOGGER.info("OOF Check - NaN count: %s, Data types: %s, Min: %s, Max: %s", blend_oof.isna().sum().to_dict(), blend_oof.dtypes.to_dict(), blend_oof.min().to_dict(), blend_oof.max().to_dict())

    LOGGER.info("Fitting LogisticRegression Meta-Model...")
    meta_model = LogisticRegression(C=0.1, penalty='l2', random_state=RANDOM_STATE)
    meta_model.fit(blend_oof, y_train)
    meta_probabilities = meta_model.predict_proba(blend_oof)[:, 1]

    def optimize_threshold(oof_probs, y_true):
        """
        Scans thresholds between 0.40 and 0.60 on OOF predictions
        to find the decision boundary that maximizes Local CV Accuracy.
        """
        best_threshold = 0.50
        best_score = 0.0
        thresholds = np.linspace(0.40, 0.60, 100)

        for t in thresholds:
            preds = (oof_probs >= t).astype(int)
            score = accuracy_score(y_true, preds)
            if score > best_score:
                best_score = score
                best_threshold = t

        LOGGER.info(f"🏆 Optimal decision threshold found on OOF: {best_threshold:.4f} (Accuracy: {best_score:.4f})")
        return best_threshold

    optimal_threshold = optimize_threshold(meta_probabilities, y_train)

    stack_metrics = _metrics(y_train, meta_probabilities)
    stack_metrics["training_metrics"] = _metrics(y_train, meta_probabilities) # Optimization ran on full OOF directly
    LOGGER.info("Meta-Model OOF metrics: %s", stack_metrics)

    test_base_initial = pd.DataFrame({
        name: full_models[name].predict_proba(X_test)[:, 1] for name in blend_models
    })

    test_predictions_initial = meta_model.predict_proba(test_base_initial)[:, 1]

    # Semi-Supervised Pseudo-Labeling via Confident Sinkhorn Allocation
    LOGGER.info("Running Confident Sinkhorn Allocation on test probabilities...")
    allocator = ConfidentSinkhornAllocator()
    high_conf_idx, pseudo_labels = allocator.fit_allocate(test_predictions_initial)
    LOGGER.info("Extracted %d high-confidence pseudo-labels.", len(high_conf_idx))

    # Augment training data in-memory
    X_train_aug = pd.concat([X_train, X_test.iloc[high_conf_idx]], ignore_index=True)
    y_train_aug = pd.concat([y_train, pd.Series(pseudo_labels[high_conf_idx])], ignore_index=True)

    # Retrain full base models on augmented dataset (with IPW)
    LOGGER.info("Retraining base models on augmented dataset...")
    retrained_models: Dict[str, Pipeline] = {}

    # Supported IPW models as defined in modeling.py
    supported_models = ['CatBoost', 'XGBoost', 'RandomForest', 'LightGBM']

    for name, estimator in models.items():
        fitted = _pipeline(clone(estimator), X_train_aug)

        if name in supported_models:
            weights = compute_ipw_weights(X_train_aug)
            fitted.fit(X_train_aug, y_train_aug, model__sample_weight=weights)
        else:
            fitted.fit(X_train_aug, y_train_aug)

        retrained_models[name] = fitted

    # Predict on the entire test set again with the retrained models
    test_base_retrained = pd.DataFrame({
        name: retrained_models[name].predict_proba(X_test)[:, 1] for name in blend_models
    }) if blend_models else pd.DataFrame({
        name: model.predict_proba(X_test)[:, 1] for name, model in retrained_models.items()
    })

    # Use the FROZEN Meta-Model to combine predictions
    test_predictions_final = meta_model.predict_proba(test_base_retrained)[:, 1]

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        TARGET_COLUMN: (test_predictions_final >= optimal_threshold).astype(int),
    })

    submission_path = Path(SUBMISSIONS_DIR) / "submission_stacking.csv"
    submission.to_csv(submission_path, index=False)

    submission_prob = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        TARGET_COLUMN: test_predictions_final,
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
    # Save the optimal threshold so final_submission can use it
    summary_path = Path(SUBMISSIONS_DIR) / "submission_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["optimal_threshold"] = optimal_threshold
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    for name, model in retrained_models.items():
        joblib.dump(model, Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib")
    joblib.dump(meta_model, Path(MODELS_DIR) / "stacking_meta_model.joblib")
    LOGGER.info("Stacking submission saved to %s", submission_path)
    return submission


if __name__ == "__main__":
    run_stacking_pipeline()
