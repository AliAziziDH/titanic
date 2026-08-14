"""Model comparison, validation, and submission generation for Titanic."""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict

from gplearn.genetic import SymbolicTransformer
from tabpfn import TabPFNClassifier

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
    "Age", "SibSp", "Parch", "Family_Size", "Ticket_Frequency", "AdjFare", "WCG_Survival",
]
CATEGORICAL_FEATURES = [
    "Sex", "Embarked", "Title_Num", "Title_Encoded", "Deck_Num", "Deck_Encoded",
    "Deck", "Deck_Group", "Family_Name", "Last_Name", "Family_Size_Category", "Ticket_Prefix",
    "AdjFare_Bin", "Age_Band", "Sex_Pclass", "Title_Sex", "Group_ID",
]
BINARY_FEATURES = ["Has_Cabin", "Is_Alone", "Is_Group", "Is_Mother", "WCG_Member"]


class WCGSurvivalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.default_rate = 0.5
        self.pass1_groups = {}
        self.pass2_groups = {}

    def fit(self, X, y=None):
        if y is None or 'Last_Name' not in X or 'AdjFare' not in X or 'Ticket' not in X:
            return self

        df = X.copy()
        df['Survived'] = y

        # We store lists of (index, survival) for Pass 1 and Pass 2

        df['Pass1_Group'] = df['Last_Name'].astype(str) + "_" + df['AdjFare'].astype(str)
        df['Pass2_Group'] = df['Ticket'].astype(str)

        self.pass1_groups = df.groupby('Pass1_Group').apply(
            lambda g: list(zip(g.index, g['Survived'])), include_groups=False
        ).to_dict()

        self.pass2_groups = df.groupby('Pass2_Group').apply(
            lambda g: list(zip(g.index, g['Survived'])), include_groups=False
        ).to_dict()

        return self

    def transform(self, X):
        df = X.copy()
        if 'Last_Name' not in df or 'AdjFare' not in df or 'Ticket' not in df:
            df['WCG_Survival'] = self.default_rate
            return df

        df['Pass1_Group'] = df['Last_Name'].astype(str) + "_" + df['AdjFare'].astype(str)
        df['Pass2_Group'] = df['Ticket'].astype(str)

        def calculate_survival(row):
            # Pass 1
            pass1_members = self.pass1_groups.get(row['Pass1_Group'], [])
            # Exclude self
            other_pass1 = [surv for idx, surv in pass1_members if idx != row.name]

            if len(other_pass1) > 0:
                max_surv = max(other_pass1)
                min_surv = min(other_pass1)
                if max_surv == 1.0:
                    return 1.0
                if min_surv == 0.0:
                    return 0.0

            # Pass 2
            pass2_members = self.pass2_groups.get(row['Pass2_Group'], [])
            # Exclude self
            other_pass2 = [surv for idx, surv in pass2_members if idx != row.name]

            if len(other_pass2) > 0:
                max_surv = max(other_pass2)
                min_surv = min(other_pass2)
                if max_surv == 1.0:
                    return 1.0
                if min_surv == 0.0:
                    return 0.0

            return self.default_rate

        df['WCG_Survival'] = df.apply(calculate_survival, axis=1)

        # Clean up temporary columns
        df.drop(columns=['Pass1_Group', 'Pass2_Group'], inplace=True, errors='ignore')

        return df


class AgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imputer = None
        self.available_cols = []
        self.sex_map = False

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        self.imputer = IterativeImputer(random_state=self.random_state)

        # Features to conditionally group by / use for imputing
        cols_to_use = ['Age', 'Pclass', 'Title_Encoded']
        X_temp = X.copy()
        if 'Sex' in X_temp.columns:
            if X_temp['Sex'].dtype == 'O' or X_temp['Sex'].dtype == 'string':
                self.sex_map = True
                X_temp['Sex'] = X_temp['Sex'].map({'male': 0, 'female': 1})
            cols_to_use.append('Sex')

        # Only fit on available cols
        self.available_cols = [c for c in cols_to_use if c in X_temp.columns]

        if len(self.available_cols) > 0 and 'Age' in self.available_cols:
            self.imputer.fit(X_temp[self.available_cols])
        else:
            self.imputer = None

        return self

    def transform(self, X):
        df = X.copy()
        if self.imputer is not None and 'Age' in df.columns:
            if self.sex_map and 'Sex' in df.columns:
                df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
            # fill missing
            imputed = self.imputer.transform(df[self.available_cols])
            df['Age'] = imputed[:, self.available_cols.index('Age')]
            if self.sex_map and 'Sex' in X.columns:
                df['Sex'] = X['Sex'] # restore original
        return df


class TabPFNFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tabpfn = TabPFNClassifier(n_estimators=1, ignore_pretraining_limits=True,
            model_path="models/tabpfn/tabpfn-v3-classifier-v3_default.ckpt",
            device='cpu'
        )
        self.fitted_ = False
        self.train_probs_ = None

    def fit(self, X, y=None):
        if y is None:
            return self

        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        X_imputed = np.nan_to_num(X, nan=-999.0)

        self.train_probs_ = cross_val_predict(
            self.tabpfn, X_imputed, y, cv=cv, method='predict_proba', n_jobs=-1
        )[:, 1].reshape(-1, 1)

        self.tabpfn.fit(X_imputed, y)
        self.fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.train_probs_

    def transform(self, X):
        if not self.fitted_:
            return np.zeros((X.shape[0], 1))

        X_imputed = np.nan_to_num(X, nan=-999.0)
        probs = self.tabpfn.predict_proba(X_imputed)[:, 1].reshape(-1, 1)
        return probs


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        from scipy import sparse
        if sparse.issparse(X):
            return X.toarray()
        return np.array(X)


class PipelineWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, union):
        self.preprocessor = preprocessor
        self.union = union
        from sklearn.pipeline import Pipeline
        self.pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("meta_union", union)
        ])

    def fit(self, X, y=None):
        self.pipe.fit(X, y)
        return self

    def transform(self, X):
        return self.pipe.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipe.fit_transform(X, y)


def build_meta_features(preprocessor):
    symbolic = SymbolicTransformer(
        population_size=100,
        hall_of_fame=20,
        n_components=10,
        generations=5,
        tournament_size=10,
        stopping_criteria=1.0,
        const_range=(-1.0, 1.0),
        init_depth=(2, 4),
        init_method='half and half',
        function_set=['add', 'sub', 'mul', 'div', 'sqrt'],
        metric='pearson',
        parsimony_coefficient=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        feature_names=None,
        warm_start=False,
        low_memory=False,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE
    )

    from sklearn.compose import ColumnTransformer
    # Age is 0, Family_Size is 3, AdjFare is 5 based on numeric pipeline output
    # Also pass only these features to TabPFN
    core_features = ColumnTransformer(
        [("num_features", "passthrough", [0, 3, 5])],
        remainder="drop"
    )
    from sklearn.pipeline import make_pipeline
    union = FeatureUnion([
        ("original", FunctionTransformer()),
        ("symbolic", make_pipeline(core_features, symbolic)),
        ("tabpfn", make_pipeline(core_features, TabPFNFeatureExtractor(random_state=RANDOM_STATE)))
    ])

    union = make_pipeline(ToDenseTransformer(), union)
    return PipelineWrapper(preprocessor, union)


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
    numeric_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = ImbPipeline([
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
            iterations=500, learning_rate=0.05, depth=4,
            l2_leaf_reg=50.0, subsample=0.7,
            random_seed=RANDOM_STATE, verbose=False, task_type="CPU",
        )
    except ImportError:
        LOGGER.warning("CatBoost is unavailable; skipping it")
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            reg_lambda=50.0, subsample=0.7, colsample_bytree=0.7,
            random_state=RANDOM_STATE, tree_method="hist", eval_metric="logloss",
        )
    except ImportError:
        LOGGER.warning("XGBoost is unavailable; skipping it")
    try:
        from lightgbm import LGBMClassifier

        params = dict(LIGHTGBM_PARAMS)
        params.update({
            "device": "cpu", "verbosity": -1,
            "max_depth": 4, "reg_lambda": 50.0,
            "subsample": 0.7, "colsample_bytree": 0.7
        })
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
        pipeline = ImbPipeline([
            ("wcg_encoder", WCGSurvivalEncoder()),
            ("age_imputer", AgeImputer(random_state=RANDOM_STATE)),
            ("meta_features", build_meta_features(build_preprocessor(X_train))),
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

    # Train and save ALL individual models so final_submission.py can load them
    models_dict = default_models()
    for name, model in models_dict.items():
        pipeline = ImbPipeline([
            ("wcg_encoder", WCGSurvivalEncoder()),
            ("age_imputer", AgeImputer(random_state=RANDOM_STATE)),
            ("meta_features", build_meta_features(build_preprocessor(X_train))),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / f"{name.lower()}_final.joblib")

    # The rest proceeds as before for the "best" model logic
    best_model = models_dict[best["model"]]
    final_pipeline = ImbPipeline([
        ("wcg_encoder", WCGSurvivalEncoder()),
        ("age_imputer", AgeImputer(random_state=RANDOM_STATE)),
        ("meta_features", build_meta_features(build_preprocessor(X_train))),
        ("model", best_model),
    ])
    final_pipeline.fit(X_train, y_train)
    test_predictions = final_pipeline.predict(X_test).astype(int)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], TARGET_COLUMN: test_predictions})
    submission_path = Path(SUBMISSIONS_DIR) / "submission_modeling.csv"
    submission.to_csv(submission_path, index=False)
    (MODEL_DIR / "best_model.json").write_text(
        json.dumps({"model": best["model"], "metrics": best["metrics"]}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Selected %s; submission saved to %s", best["model"], submission_path)
    return submission


if __name__ == "__main__":
    run_modeling_pipeline()
