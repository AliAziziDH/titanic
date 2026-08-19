"""Model interpretation, sanity checks, and error analysis."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATA_PROCESSED_DIR, MODELS_DIR

LOGGER = logging.getLogger("titanic.interpret")
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _load_model() -> Any:
    """Load the preferred saved model artifact."""
    stacking_path = Path(MODELS_DIR) / "stacking_meta_model.joblib"
    catboost_path = Path(MODELS_DIR) / "catboost_final.joblib"
    if catboost_path.exists():
        return joblib.load(catboost_path)
    if stacking_path.exists():
        LOGGER.warning("Using stacking meta-model; raw-feature SHAP is unavailable for this artifact")
        return joblib.load(stacking_path)
    raise FileNotFoundError("No saved CatBoost or stacking model found")


def _shap_values(model: Any, frame: pd.DataFrame, sample_size: int = 500) -> Tuple[Any, pd.DataFrame, list[str]]:
    """Compute SHAP values for a fitted sklearn pipeline."""
    try:
        import shap
    except ImportError as error:
        raise ImportError("Install shap to run model interpretation") from error
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps:
        raise ValueError("Interpretation requires a fitted preprocessing pipeline")
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    sample = frame.sample(min(sample_size, len(frame)), random_state=42)
    transformed = preprocessor.transform(sample)
    names = list(preprocessor.get_feature_names_out())
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    return values, sample, names


def _save_shap_plots(
    values: Any, transformed: pd.DataFrame, names: list[str], frame: pd.DataFrame, explainer: Any
) -> Dict[str, Any]:
    import shap

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    top_features = [
        {"feature": names[index], "mean_abs_shap": float(importance[index])}
        for index in order[:10]
    ]
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()
    for feature in ["Sex", "Title", "Title_Num", "Title_Encoded", "Fare_per_Person", "Pclass", "Age"]:
        matches = [index for index, name in enumerate(names) if feature.lower() in name.lower()]
        if not matches:
            continue
        shap.dependence_plot(matches[0], values, transformed, feature_names=names, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_dependence_{feature.lower()}.png", dpi=150)
        plt.close()
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]
    for sample_index in range(min(3, len(transformed))):
        explanation = shap.Explanation(
            values=values[sample_index],
            base_values=float(expected_value),
            data=transformed.iloc[sample_index].to_numpy(),
            feature_names=names,
        )
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{sample_index + 1}.png", dpi=150)
        plt.close()
    return {"top_features": top_features}


def _sanity_checks(
    top_features: list[dict[str, Any]], names: list[str], values: Any, transformed: pd.DataFrame
) -> Dict[str, Any]:
    ranked = [item["feature"].lower() for item in top_features]
    top_expected = False
    passenger_id_high = False
    for feature in ranked[:5]:
        if not top_expected and ("sex" in feature or "title" in feature or "pclass" in feature):
            top_expected = True
        if not passenger_id_high and "passengerid" in feature:
            passenger_id_high = True
        if top_expected and passenger_id_high:
            break
    checks = {
        "expected_features_rank_high": {"passed": top_expected},
        "passenger_id_not_high": {"passed": not passenger_id_high},
    }
    def mean_effect(matching: list[str]) -> float | None:
        indices = [index for index, name in enumerate(names) if any(key in name.lower() for key in matching)]
        return float(values[:, indices].mean()) if indices else None

    female_effect = mean_effect(["sex_female"])
    mr_effect = mean_effect(["title_mr"])
    pclass_indices = [index for index, name in enumerate(names) if name.lower().endswith("pclass")]
    pclass_direction = None
    if pclass_indices:
        pclass_direction = float(np.corrcoef(transformed.iloc[:, pclass_indices[0]], values[:, pclass_indices[0]])[0, 1])
    checks["female_positive_effect"] = {
        "passed": female_effect is not None and female_effect > 0,
        "mean_shap": female_effect,
    }
    checks["mr_negative_effect"] = {
        "passed": mr_effect is not None and mr_effect < 0,
        "mean_shap": mr_effect,
    }
    checks["pclass_higher_value_negative_effect"] = {
        "passed": pclass_direction is not None and pclass_direction < 0,
        "correlation": pclass_direction,
    }
    LOGGER.info("Sanity check expected features rank high: %s", checks["expected_features_rank_high"]["passed"])
    LOGGER.info("Sanity check PassengerId not high: %s", checks["passenger_id_not_high"]["passed"])
    return checks


def _error_analysis(model: Any, data: pd.DataFrame) -> Dict[str, Any]:
    features = [column for column in data.columns if column not in {"Survived", "PassengerId"}]
    predictions = model.predict(data[features]).astype(int)
    errors = data.loc[predictions != data["Survived"]].copy()
    errors["Predicted"] = predictions[predictions != data["Survived"]]
    errors.to_csv(REPORTS_DIR / "misclassified_samples.csv", index=False)
    summary: Dict[str, Any] = {"count": int(len(errors)), "groups": {}}
    for column in ["Pclass", "Sex", "Age_Band", "Embarked"]:
        if column not in data:
            continue
        grouped = data.assign(_error=(predictions != data["Survived"])).groupby(column, dropna=False)["_error"].agg(
            error_rate="mean", count="size", errors="sum"
        )
        grouped["error_rate"] = grouped["error_rate"].round(4)
        summary["groups"][column] = grouped.reset_index().to_dict(orient="records")
        LOGGER.info("Highest %s error groups:\n%s", column, grouped.sort_values("error_rate", ascending=False).head(3))
    return summary


def run_interpretation() -> Dict[str, Any]:
    """Generate interpretation reports for the saved final model."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(DATA_PROCESSED_DIR) / "train_clean.csv")
    model = _load_model()
    features = [column for column in train.columns if column not in {"Survived", "PassengerId"}]
    try:
        values, sample, names = _shap_values(model, train[features])
        preprocessed = model.named_steps["preprocessor"].transform(sample)
        if hasattr(preprocessed, "toarray"):
            preprocessed = preprocessed.toarray()
        transformed = pd.DataFrame(preprocessed, columns=names)
        import shap

        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_report = _save_shap_plots(values, transformed, names, sample, explainer)
        sanity = _sanity_checks(shap_report["top_features"], names, values, transformed)
    except ImportError as error:
        LOGGER.warning("SHAP analysis skipped: %s", error)
        shap_report = {"top_features": [], "skipped": True}
        sanity = {"skipped": True}
    errors = _error_analysis(model, train)
    report = {"top_features": shap_report["top_features"], "sanity_checks": sanity, "error_analysis": errors}
    (REPORTS_DIR / "interpretation_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Interpretation summary saved to %s", REPORTS_DIR / "interpretation_summary.json")
    return report


if __name__ == "__main__":
    run_interpretation()
