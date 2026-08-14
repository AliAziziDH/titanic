"""Generate final Titanic submissions from saved models and ensembles."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, MODELS_DIR, TARGET_COLUMN, get_input_dir
from src.modeling import PipelineWrapper, TabPFNFeatureExtractor, ToDenseTransformer, WCGSurvivalEncoder, AgeImputer # Required for unpickling models

LOGGER = logging.getLogger("titanic.final_submission")
PROJECT_DIR = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"


def _load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed data, rebuilding it from local or Kaggle raw data when needed."""
    train_path = Path(DATA_PROCESSED_DIR) / "train_clean.csv"
    test_path = Path(DATA_PROCESSED_DIR) / "test_clean.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)
    from src.features import engineer_features
    from src.imputation import impute_missing_values, save_clean_data

    input_dir = get_input_dir()
    raw_train = pd.read_csv(input_dir / "train.csv")
    raw_test = pd.read_csv(input_dir / "test.csv")
    engineered_train = engineer_features(raw_train, reference_df=pd.concat([raw_train, raw_test], ignore_index=True))
    engineered_test = engineer_features(raw_test, reference_df=pd.concat([raw_train, raw_test], ignore_index=True))
    clean_train, clean_test = impute_missing_values(engineered_train, engineered_test)
    save_clean_data(clean_train, clean_test)
    return clean_train, clean_test


class WCGPostProcessor:
    """Deterministic post-processor based on Woman-Child-Group survival."""

    def __init__(self) -> None:
        self.pass1_groups = {}
        self.pass2_groups = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        # We need raw data for 'Ticket' and 'Last_Name' if they are missing
        # But 'Ticket' is not dropped in engineering unless specifically configured. Wait, 'Ticket' IS dropped from model features.
        from src.config import get_input_dir
        input_dir = get_input_dir()
        raw_train = pd.read_csv(input_dir / "train.csv")

        df = train_df.copy()
        if 'Ticket' not in df:
            df['Ticket'] = raw_train['Ticket']
        if 'Last_Name' not in df:
            df['Last_Name'] = raw_train['Name'].str.split(",", n=1).str[0].str.strip()

        # AdjFare is already in train_df because it's engineered.
        # But just in case, we compute it if missing.
        if 'AdjFare' not in df:
             ticket_counts = df['Ticket'].value_counts(dropna=False)
             df['AdjFare'] = raw_train['Fare'] / df['Ticket'].map(ticket_counts).fillna(1).replace(0, 1)

        df['Pass1_Group'] = df['Last_Name'].astype(str) + "_" + df['AdjFare'].astype(str)
        df['Pass2_Group'] = df['Ticket'].astype(str)

        self.pass1_groups = df.groupby('Pass1_Group')['Survived'].apply(list).to_dict()
        self.pass2_groups = df.groupby('Pass2_Group')['Survived'].apply(list).to_dict()

    def transform(self, test_df: pd.DataFrame, baseline_probs: np.ndarray) -> np.ndarray:
        final_probs = baseline_probs.copy()

        from src.config import get_input_dir
        input_dir = get_input_dir()
        raw_test = pd.read_csv(input_dir / "test.csv")

        # We need raw_train for exact ticket counts to compute AdjFare perfectly if missing.
        raw_train = pd.read_csv(input_dir / "train.csv")
        combined = pd.concat([raw_train, raw_test], ignore_index=True)

        df = test_df.copy()
        if 'Ticket' not in df:
            df['Ticket'] = raw_test['Ticket']
        if 'Last_Name' not in df:
            df['Last_Name'] = raw_test['Name'].str.split(",", n=1).str[0].str.strip()

        if 'AdjFare' not in df:
             ticket_counts = combined['Ticket'].value_counts(dropna=False)
             df['AdjFare'] = raw_test['Fare'] / df['Ticket'].map(ticket_counts).fillna(1).replace(0, 1)

        df['Pass1_Group'] = df['Last_Name'].astype(str) + "_" + df['AdjFare'].astype(str)
        df['Pass2_Group'] = df['Ticket'].astype(str)

        for idx in range(len(df)):
            if df.iloc[idx]["WCG_Member"] == 1:
                row = df.iloc[idx]
                pass1_members = self.pass1_groups.get(row['Pass1_Group'], [])
                if len(pass1_members) > 0:
                    if max(pass1_members) == 1.0:
                        final_probs[idx] = 1.0
                        continue
                    if min(pass1_members) == 0.0:
                        final_probs[idx] = 0.0
                        continue
                pass2_members = self.pass2_groups.get(row['Pass2_Group'], [])
                if len(pass2_members) > 0:
                    if max(pass2_members) == 1.0:
                        final_probs[idx] = 1.0
                        continue
                    if min(pass2_members) == 0.0:
                        final_probs[idx] = 0.0
                        continue
        return final_probs

def _load_scores() -> Dict[str, Dict[str, float]]:
    """Load ROC-AUC and accuracy scores for weighting and reporting."""
    scores: Dict[str, Dict[str, float]] = {}
    path = Path(EXPERIMENTS_DIR) / "cv_results.json"
    if path.exists():
        for result in json.loads(path.read_text(encoding="utf-8")):
            metrics = result.get("metrics", {})
            scores[result["model"]] = {
                key: float(metrics[key]["mean"])
                for key in ("accuracy", "roc_auc")
                if key in metrics
            }
    stacking_path = Path(EXPERIMENTS_DIR) / "stacking_results.json"
    if stacking_path.exists():
        result = json.loads(stacking_path.read_text(encoding="utf-8"))
        scores["Stacking"] = {
            key: float(result["stacker"][key])
            for key in ("accuracy", "roc_auc")
            if key in result.get("stacker", {})
        }
    return scores


def _write_submission(name: str, passenger_ids: pd.Series, probabilities: np.ndarray) -> Path:
    if len(passenger_ids) != 418:
        raise ValueError(f"CRITICAL ERROR: Refusing to generate submission. Test data has {len(passenger_ids)} rows. Must be exactly 418!")
    if not (passenger_ids.min() == 892 and passenger_ids.max() == 1309):
        raise ValueError(f"CRITICAL ERROR: PassengerId range is {passenger_ids.min()}-{passenger_ids.max()}. Must be exactly 892-1309!")

    path = SUBMISSIONS_DIR / f"submission_{name.lower()}.csv"
    pd.DataFrame({
        "PassengerId": passenger_ids,
        TARGET_COLUMN: (np.asarray(probabilities) >= 0.5).astype(int),
    }).to_csv(path, index=False)
    return path


def _load_individual_models() -> Dict[str, Any]:
    candidates = {
        "CatBoost": "catboost_final.joblib",
        "XGBoost": "xgboost_final.joblib",
        "LightGBM": "lightgbm_final.joblib",
        "RandomForest": "randomforest_final.joblib",
    }
    loaded = {}
    for name, filename in candidates.items():
        path = Path(MODELS_DIR) / filename
        if path.exists():
            loaded[name] = joblib.load(path)
        else:
            LOGGER.info("Optional individual model not found: %s", path)
    return loaded


def _load_stacking_models() -> tuple[Dict[str, Any], Any]:
    order = ["RandomForest", "MLP", "CatBoost", "LightGBM", "XGBoost"]
    base = {}
    for name in order:
        path = Path(MODELS_DIR) / f"stacking_{name.lower()}.joblib"
        if path.exists():
            base[name] = joblib.load(path)
    meta_path = Path(MODELS_DIR) / "stacking_meta_model.joblib"
    if not meta_path.exists():
        raise FileNotFoundError("Complete stacking model artifacts are not available")
    return {name: base[name] for name in order if name in base}, joblib.load(meta_path)


def run_final_submission() -> Dict[str, Any]:
    """Generate individual, stacking, weighted, and majority-vote submissions."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train, test = _load_clean_data()

    # Guardrail checking
    passenger_ids = test["PassengerId"]
    if len(test) != 418:
        raise ValueError(f"CRITICAL ERROR: Refusing to generate submission. Test data has {len(test)} rows. Must be exactly 418!")
    if not (passenger_ids.min() == 892 and passenger_ids.max() == 1309):
        raise ValueError(f"CRITICAL ERROR: PassengerId range is {passenger_ids.min()}-{passenger_ids.max()}. Must be exactly 892-1309!")

    features = [column for column in train.columns if column not in {TARGET_COLUMN, "PassengerId"}]
    X_test = test[features]
    passenger_ids = test["PassengerId"]
    probabilities: Dict[str, np.ndarray] = {}
    paths: Dict[str, str] = {}

    for name, model in _load_individual_models().items():
        probabilities[name] = model.predict_proba(X_test)[:, 1]
        paths[name] = str(_write_submission(name, passenger_ids, probabilities[name]))

    try:
        base_models, meta_model = _load_stacking_models()

        # Meta model is an array of SLSQP optimal weights
        blend_models = ["XGBoost", "LightGBM", "CatBoost"]
        blend_models = [m for m in blend_models if m in base_models.keys()]

        if isinstance(meta_model, np.ndarray): # It's weights
            if not blend_models:
                base_predictions = pd.DataFrame({
                    name: model.predict_proba(X_test)[:, 1] for name, model in base_models.items()
                })
                stack_probability = np.dot(base_predictions, meta_model)
            else:
                base_predictions = pd.DataFrame({
                    name: base_models[name].predict_proba(X_test)[:, 1] for name in blend_models
                })
                stack_probability = np.dot(base_predictions, meta_model)
        else:
            base_predictions = pd.DataFrame({
                name: model.predict_proba(X_test)[:, 1] for name, model in base_models.items()
            })
            stack_probability = meta_model.predict_proba(base_predictions)[:, 1]

        wcg_processor = WCGPostProcessor()
        wcg_processor.fit(train)
        stack_probability = wcg_processor.transform(test, stack_probability)

        probabilities["Stacking"] = stack_probability
        paths["Stacking"] = str(_write_submission("stacking", passenger_ids, stack_probability))
    except FileNotFoundError as error:
        LOGGER.warning("Skipping stacking submission: %s", error)

    scores = _load_scores()
    weighted_names = [name for name in probabilities if name in scores and name != "Stacking"]
    if weighted_names:
        weights = np.array([scores[name].get("roc_auc", scores[name]["accuracy"]) for name in weighted_names])
        weights /= weights.sum()
        weighted_probability = sum(
            weight * probabilities[name] for weight, name in zip(weights, weighted_names)
        )
        paths["WeightedEnsemble"] = str(
            _write_submission("weighted_ensemble", passenger_ids, weighted_probability)
        )

        hard_predictions = np.column_stack([
            probabilities[name] >= 0.5 for name in weighted_names
        ])
        majority_probability = (hard_predictions.sum(axis=1) >= (len(weighted_names) / 2)).astype(float)
        paths["MajorityVote"] = str(
            _write_submission("majority_vote", passenger_ids, majority_probability)
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "submissions": paths,
        "rows": int(len(test)),
    }
    summary_path = SUBMISSIONS_DIR / "submission_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Generated %d submission files in %s", len(paths), SUBMISSIONS_DIR)
    return summary


if __name__ == "__main__":
    run_final_submission()
