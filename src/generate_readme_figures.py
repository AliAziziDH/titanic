"""Generate polished, reproducible figures referenced by the project README."""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, get_input_dir

LOGGER = logging.getLogger("titanic.readme_figures")
PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
FIGURE_DPI = 300
FONT_SIZE = 12

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": 15, "axes.labelsize": FONT_SIZE})


def _load_model() -> Any:
    """Load the saved tree-based pipeline used for portfolio figures."""
    import joblib

    for filename in ("catboost_final.joblib", "stacking_catboost.joblib"):
        path = Path(MODELS_DIR) / filename
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError("No saved CatBoost pipeline was found")


def _load_training_data() -> pd.DataFrame:
    """Load clean data or rebuild it from the configured Kaggle input."""
    clean_path = Path(DATA_PROCESSED_DIR) / "train_clean.csv"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    input_dir = get_input_dir()
    raw_path = input_dir / "train.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Neither processed data nor Kaggle train.csv is available")
    from src.features import engineer_features
    from src.imputation import impute_missing_values

    raw_train = pd.read_csv(raw_path)
    raw_test = pd.read_csv(input_dir / "test.csv")
    return impute_missing_values(
        engineer_features(raw_train), engineer_features(raw_test)
    )[0]


def _load_test_data() -> pd.DataFrame:
    """Load clean test data or rebuild it from the configured Kaggle input."""
    clean_path = Path(DATA_PROCESSED_DIR) / "test_clean.csv"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    from src.features import engineer_features
    from src.imputation import impute_missing_values

    input_dir = get_input_dir()
    raw_train = pd.read_csv(input_dir / "train.csv")
    raw_test = pd.read_csv(input_dir / "test.csv")
    return impute_missing_values(
        engineer_features(raw_train), engineer_features(raw_test)
    )[1]


def _save_figure(name: str, figure: plt.Figure) -> None:
    """Save one completed figure at portfolio-quality resolution."""
    path = FIGURES_DIR / name
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    LOGGER.info("Saved %s", path)


def _clean_feature_name(name: Any) -> str:
    """Make transformed pipeline feature names readable in charts."""
    value = str(name).replace("num__", "").replace("cat__", "")
    value = value.replace("_", " ")
    return " ".join(value.split()).title()


def _feature_importance_figure(model: Any, X: pd.DataFrame) -> Optional[plt.Figure]:
    """Create a finite, annotated top-feature importance chart."""
    if not hasattr(model, "named_steps"):
        LOGGER.warning("Feature importance skipped: model is not a pipeline")
        return None
    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None or not hasattr(estimator, "feature_importances_"):
        LOGGER.warning("Feature importance skipped: estimator has no built-in importances")
        return None
    names = list(preprocessor.get_feature_names_out())
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    count = min(len(names), len(importances))
    importance = pd.DataFrame({
        "feature": [_clean_feature_name(name) for name in names[:count]],
        "importance": importances[:count],
    })
    importance = importance.replace([np.inf, -np.inf], np.nan).dropna(subset=["importance"])
    importance = importance[importance["feature"].str.lower().ne("nan")]
    importance = importance.sort_values("importance", ascending=False).head(15)
    if importance.empty:
        LOGGER.warning("Feature importance skipped: no finite values")
        return None
    importance = importance.sort_values("importance")
    figure, axis = plt.subplots(figsize=(10, 7))
    bars = axis.barh(importance["feature"], importance["importance"], color="#3478b9")
    axis.set_title("Top 15 Feature Importances - CatBoost", pad=14, fontweight="bold")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    offset = max(float(importance["importance"].max()) * 0.015, 1e-6)
    for bar, value in zip(bars, importance["importance"]):
        axis.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=10,
        )
    axis.set_xlim(0, float(importance["importance"].max()) * 1.18)
    return figure


def _validation_predictions(
    model: Any, X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load OOF stacker predictions when available, otherwise use model predictions."""
    oof_path = PROJECT_DIR / "experiments" / "oof_predictions.npz"
    meta_path = Path(MODELS_DIR) / "stacking_meta_model.joblib"
    if oof_path.exists() and meta_path.exists():
        import joblib

        oof = np.load(oof_path)
        probabilities = joblib.load(meta_path).predict_proba(oof["predictions"])[:, 1]
        targets = oof["target"].astype(int)
    else:
        probabilities = model.predict_proba(X)[:, 1]
        targets = y.to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    return targets, predictions, probabilities


def _confusion_matrix_figure(
    targets: np.ndarray, predictions: np.ndarray
) -> plt.Figure:
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    percentages = matrix / matrix.sum() * 100
    annotations = np.array([
        [f"{matrix[row, col]}\n({percentages[row, col]:.1f}%)" for col in range(2)]
        for row in range(2)
    ])
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="white",
        xticklabels=["Did not survive", "Survived"],
        yticklabels=["Did not survive", "Survived"],
        ax=axis,
    )
    axis.set_title("Confusion Matrix - Stacking Ensemble", pad=14, fontweight="bold")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    return figure


def _roc_figure(targets: np.ndarray, probabilities: np.ndarray) -> plt.Figure:
    auc = roc_auc_score(targets, probabilities)
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, probabilities)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(false_positive_rate, true_positive_rate, color="#d95f02", linewidth=2.5, label=f"AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Random baseline")
    axis.set_title("ROC-AUC Curve - Stacking Ensemble", pad=14, fontweight="bold")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="lower right", frameon=True)
    return figure


def _survival_comparison_figure(train_rate: float, test_rate: float) -> plt.Figure:
    data = pd.DataFrame({"Dataset": ["Training", "Predicted test"], "Survival rate": [train_rate, test_rate]})
    figure, axis = plt.subplots(figsize=(8, 6))
    bars = sns.barplot(
        data=data,
        x="Dataset",
        y="Survival rate",
        hue="Dataset",
        palette=["#3478b9", "#e67e22"],
        legend=False,
        ax=axis,
    )
    axis.set_title("Survival Rate: Training vs Predicted Test", pad=14, fontweight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("Survival rate")
    axis.set_ylim(0, 1)
    for bar, value in zip(bars.patches, data["Survival rate"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    return figure


def run() -> None:
    """Generate all available README figures without creating corrupt files."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        train = _load_training_data()
        test = _load_test_data()
        model = _load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        LOGGER.warning("Figure generation skipped: %s", error)
        return

    features = [column for column in train.columns if column not in {"Survived", "PassengerId"}]
    test_features = [column for column in test.columns if column != "PassengerId"]
    X, y = train[features], train["Survived"].astype(int)
    if features != test_features:
        LOGGER.warning("Figure generation skipped: train/test feature schemas differ")
        return

    try:
        figure = _feature_importance_figure(model, X)
        if figure is not None:
            _save_figure("feature_importance.png", figure)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        LOGGER.warning("Feature importance failed: %s", error)

    try:
        targets, predictions, probabilities = _validation_predictions(model, X, y)
        _save_figure("confusion_matrix.png", _confusion_matrix_figure(targets, predictions))
        _save_figure("roc_auc_curve.png", _roc_figure(targets, probabilities))
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        LOGGER.warning("Validation figures failed: %s", error)

    try:
        test_probability = model.predict_proba(test[test_features])[:, 1]
        figure = _survival_comparison_figure(float(y.mean()), float((test_probability >= 0.5).mean()))
        _save_figure("survival_comparison.png", figure)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        LOGGER.warning("Survival comparison failed: %s", error)

    LOGGER.info("Figure generation complete: %s", FIGURES_DIR)


if __name__ == "__main__":
    run()
