"""Optuna hyperparameter tuning for Titanic classifiers."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import optuna

from src.config import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, MODELS_DIR, RANDOM_STATE
from src.modeling import build_preprocessor

LOGGER = logging.getLogger("titanic.tuning")
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)


def _suggest_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    if model_name == "CatBoost":
        return {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "iterations": trial.suggest_int("iterations", 200, 800, step=50),
        }
    if model_name == "LightGBM":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        }
    if model_name == "XGBoost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _create_model(model_name: str, params: Dict[str, Any]) -> Any:
    common = {"random_state": RANDOM_STATE}
    if model_name == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params, **common, verbose=False, task_type="CPU")
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, **common, device="cpu", verbosity=-1)
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params, **common, tree_method="hist", eval_metric="logloss"
        )
    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params, **common, n_jobs=-1)
    raise ValueError(f"Unsupported model: {model_name}")


def _objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    params = _suggest_params(trial, model_name)
    scores = []
    for step, (train_idx, validation_idx) in enumerate(CV.split(X_train, y_train)):
        estimator = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", _create_model(model_name, params)),
        ])
        estimator.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        train_predictions = estimator.predict(X_train.iloc[train_idx])
        validation_predictions = estimator.predict(X_train.iloc[validation_idx])
        train_accuracy = accuracy_score(y_train.iloc[train_idx], train_predictions)
        validation_accuracy = accuracy_score(y_train.iloc[validation_idx], validation_predictions)
        if train_accuracy - validation_accuracy > 0.05:
            raise optuna.TrialPruned("Training/validation accuracy gap exceeded 5%")
        scores.append(validation_accuracy)
        running_score = float(np.mean(scores))
        trial.report(running_score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> optuna.Study:
    """Tune one model and persist its best parameters and trial history."""
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name=f"{model_name}_tuning"
    )
    study.optimize(
        lambda trial: _objective(trial, model_name, X_train, y_train),
        n_trials=n_trials,
        callbacks=[_stop_after_no_improvement(20)],
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_params = study.best_params if completed else {}
    best_value = study.best_value if completed else None
    (Path(MODELS_DIR) / f"{model_name.lower()}_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(
        Path(EXPERIMENTS_DIR) / f"{model_name.lower()}_trials.csv", index=False
    )
    if completed:
        LOGGER.info("%s best accuracy=%.4f params=%s", model_name, best_value, best_params)
    else:
        LOGGER.warning("%s had no completed trials; all trials were pruned", model_name)
    return study


def _stop_after_no_improvement(patience: int):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = [item for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) > patience:
            recent = completed[-patience:]
            if study.best_trial.number not in {item.number for item in recent}:
                study.stop()

    return callback


def run_tuning(n_trials: int = 30) -> Dict[str, Any]:
    """Tune all supported models and save a combined summary."""
    train = pd.read_csv(Path(DATA_PROCESSED_DIR) / "train_clean.csv")
    X_train = train.drop(columns=["Survived", "PassengerId"], errors="ignore")
    y_train = train["Survived"].astype(int)
    summaries = {}
    for model_name in ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]:
        try:
            study = tune_model(model_name, X_train, y_train, n_trials=n_trials)
            completed = [
                trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            summaries[model_name] = {
                "best_accuracy": study.best_value if completed else None,
                "best_params": study.best_params if completed else {},
                "trials": len(study.trials),
            }
        except ImportError:
            LOGGER.warning("%s is unavailable; skipping tuning", model_name)
    summary_path = Path(EXPERIMENTS_DIR) / "tuning_summary.json"
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


if __name__ == "__main__":
    run_tuning()
