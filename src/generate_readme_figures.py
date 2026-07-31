"""Generate reproducible figures referenced by the project README."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_auc_score, roc_curve

from src.config import DATA_PROCESSED_DIR, KAGGLE_INPUT_DIR, MODELS_DIR

LOGGER = logging.getLogger("titanic.readme_figures")
PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"


def _load_model():
    path = Path(MODELS_DIR) / "catboost_final.joblib"
    if not path.exists():
        path = Path(MODELS_DIR) / "stacking_catboost.joblib"
    if not path.exists():
        raise FileNotFoundError("A saved CatBoost pipeline is required for figures")
    import joblib

    return joblib.load(path)


def _load_training_data() -> pd.DataFrame:
    """Load clean data or rebuild it from the configured Kaggle input."""
    clean_path = Path(DATA_PROCESSED_DIR) / "train_clean.csv"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    raw_path = Path(KAGGLE_INPUT_DIR) / "train.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Neither processed data nor Kaggle train.csv is available")
    from src.features import engineer_features
    from src.imputation import impute_missing_values

    raw_train = pd.read_csv(raw_path)
    raw_test = pd.read_csv(Path(KAGGLE_INPUT_DIR) / "test.csv")
    engineered_train = engineer_features(raw_train)
    engineered_test = engineer_features(raw_test)
    clean_train, _ = impute_missing_values(engineered_train, engineered_test)
    return clean_train


def _load_test_data() -> pd.DataFrame:
    """Load clean test data or rebuild it from the configured Kaggle input."""
    clean_path = Path(DATA_PROCESSED_DIR) / "test_clean.csv"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    from src.features import engineer_features
    from src.imputation import impute_missing_values

    raw_train = pd.read_csv(Path(KAGGLE_INPUT_DIR) / "train.csv")
    raw_test = pd.read_csv(Path(KAGGLE_INPUT_DIR) / "test.csv")
    engineered_train = engineer_features(raw_train)
    engineered_test = engineer_features(raw_test)
    _, clean_test = impute_missing_values(engineered_train, engineered_test)
    return clean_test


def run() -> None:
    """Create feature importance, confusion matrix, ROC, and domain plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = _load_training_data()
    test = _load_test_data()
    features = [column for column in train.columns if column not in {"Survived", "PassengerId"}]
    X, y = train[features], train["Survived"].astype(int)
    model = _load_model()
    predictions = model.predict(X).astype(int)
    probabilities = model.predict_proba(X)[:, 1]

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    if hasattr(estimator, "feature_importances_"):
        importance = pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        sns.barplot(x=importance.values, y=importance.index, color="steelblue")
        plt.title("Top transformed feature importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
        plt.close()

    # Prefer honest OOF predictions when the stacking run has saved them.
    oof_path = Path(PROJECT_DIR) / "experiments" / "oof_predictions.npz"
    if oof_path.exists():
        oof = __import__("numpy").load(oof_path)
        oof_predictions = oof["predictions"]
        oof_target = oof["target"].astype(int)
        meta_path = Path(MODELS_DIR) / "stacking_meta_model.joblib"
        if meta_path.exists():
            import joblib

            meta_model = joblib.load(meta_path)
            validation_probabilities = meta_model.predict_proba(oof_predictions)[:, 1]
            validation_predictions = (validation_probabilities >= 0.5).astype(int)
            validation_y = oof_target
        else:
            validation_probabilities = probabilities
            validation_predictions = predictions
            validation_y = y
    else:
        validation_probabilities = probabilities
        validation_predictions = predictions
        validation_y = y

    matrix = confusion_matrix(validation_y, validation_predictions)
    ConfusionMatrixDisplay(matrix, display_labels=["Did not survive", "Survived"]).plot(cmap="Blues")
    plt.title("Training confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(validation_y, validation_probabilities)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label=f"OOF ROC-AUC = {roc_auc_score(validation_y, validation_probabilities):.3f}")
    plt.title("OOF ROC-AUC curve")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_auc_curve.png", dpi=150)
    plt.close()

    test_features = [column for column in test.columns if column != "PassengerId"]
    test_probability = model.predict_proba(test[test_features])[:, 1]
    plot_data = pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "Survival rate": [float(y.mean()), float((test_probability >= 0.5).mean())],
    })
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="Dataset", y="Survival rate", hue="Dataset", legend=False)
    plt.ylim(0, 1)
    plt.title("Training vs predicted test survival rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "survival_comparison.png", dpi=150)
    plt.close()
    LOGGER.info("Generated README figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    run()
