"""Exploratory data analysis for the Titanic project."""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import KAGGLE_INPUT_DIR, LOGGER, TARGET_COLUMN


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"

sns.set_theme(style="whitegrid")


def _save_plot(name: str, plotter: Callable[[], None]) -> None:
    """Generate and save one figure without stopping the complete EDA run."""
    try:
        plt.figure(figsize=(9, 6))
        plotter()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
    # Plotting failures are isolated so one problematic visualization does not
    # prevent the remaining analysis and summary from being produced.
    except Exception as error:
        LOGGER.exception("Could not generate figure %s: %s", name, error)
    finally:
        plt.close("all")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original train and test datasets from Kaggle input."""
    input_dir = KAGGLE_INPUT_DIR
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train.csv and test.csv in {input_dir}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _extract_title(name: Any) -> str:
    match = re.search(r",\s*([^.]*)\.", str(name))
    return match.group(1).strip() if match else "Unknown"


def _extract_deck(cabin: Any) -> str:
    value = str(cabin).strip()
    return value[0] if value and value.lower() != "nan" else "Unknown"


def _ticket_prefix(ticket: Any) -> str:
    value = re.sub(r"\d", "", str(ticket)).replace(".", "").replace("/", "")
    return re.sub(r"\s+", "", value).upper() or "NUMERIC"


def _prepare_analysis_data(train: pd.DataFrame) -> pd.DataFrame:
    data = train.copy()
    data["FamilySize"] = data["SibSp"].fillna(0) + data["Parch"].fillna(0) + 1
    data["Is_Alone"] = (data["FamilySize"] == 1).astype(int)
    data["Title"] = data["Name"].map(_extract_title)
    data["Deck"] = data["Cabin"].map(_extract_deck)
    data["TicketPrefix"] = data["Ticket"].map(_ticket_prefix)
    data["FarePerPerson"] = data["Fare"] / data["Ticket"].map(data["Ticket"].value_counts())
    return data


def _missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    missing = data.isna().sum().to_frame("count")
    missing["percentage"] = (missing["count"] / len(data) * 100).round(2)
    return missing[missing["count"] > 0].sort_values("percentage", ascending=False)


def _iqr_outliers(data: pd.DataFrame, column: str) -> Dict[str, Any]:
    values = data[column].dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values[(values < lower) | (values > upper)]
    return {
        "count": int(outliers.size),
        "percentage": round(float(outliers.size / len(values) * 100), 2),
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
    }


def _distribution_plot(data: pd.DataFrame, column: str, name: str) -> None:
    _save_plot(name, lambda: sns.histplot(data=data, x=column, kde=True))


def _count_plot(data: pd.DataFrame, column: str, name: str, hue: Optional[str] = None) -> None:
    def plot() -> None:
        sns.countplot(data=data, x=column, hue=hue)
        plt.xticks(rotation=30)

    _save_plot(name, plot)


def _survival_rate_table(data: pd.DataFrame, column: str) -> Dict[str, float]:
    return {
        str(key): round(float(value), 4)
        for key, value in data.groupby(column, dropna=False)[TARGET_COLUMN].mean().items()
    }


def _run_univariate(data: pd.DataFrame, summary: Dict[str, Any]) -> None:
    summary["target_distribution"] = data[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict()
    _count_plot(data, TARGET_COLUMN, "target_distribution")

    numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    summary["numeric_statistics"] = data[numeric].describe().round(4).to_dict()
    summary["outliers"] = {column: _iqr_outliers(data, column) for column in ["Age", "Fare"]}
    for column in numeric:
        _distribution_plot(data, column, f"{column.lower()}_distribution")
        _save_plot(
            f"{column.lower()}_boxplot",
            lambda column=column: sns.boxplot(data=data, x=column),
        )
    for column in ["Sex", "Pclass", "Embarked"]:
        summary[f"{column.lower()}_distribution"] = (
            data[column].value_counts(normalize=True, dropna=False).round(4).to_dict()
        )
        _count_plot(data, column, f"{column.lower()}_distribution")
    summary["ticket_frequency"] = data["Ticket"].value_counts().head(20).to_dict()
    summary["cabin_frequency"] = data["Cabin"].value_counts(dropna=False).head(20).to_dict()


def _run_bivariate(data: pd.DataFrame, summary: Dict[str, Any]) -> None:
    for column in ["Sex", "Pclass", "Embarked", "Deck", "Title", "FamilySize"]:
        summary[f"survival_by_{column.lower()}"] = _survival_rate_table(data, column)
        _save_plot(
            f"survival_by_{column.lower()}",
            lambda column=column: sns.barplot(data=data, x=column, y=TARGET_COLUMN, errorbar=None),
        )
    _save_plot(
        "age_vs_survival",
        lambda: sns.boxplot(data=data, x=TARGET_COLUMN, y="Age"),
    )
    _save_plot(
        "age_vs_survival_by_title",
        lambda: sns.boxplot(data=data, x="Title", y="Age", hue=TARGET_COLUMN),
    )
    _save_plot(
        "fare_vs_survival_by_pclass_embarked",
        lambda: sns.boxplot(data=data, x="Pclass", y="Fare", hue="Embarked"),
    )
    _save_plot(
        "family_size_vs_survival",
        lambda: sns.barplot(data=data, x="FamilySize", y=TARGET_COLUMN, errorbar=None),
    )
    numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize", TARGET_COLUMN]
    summary["correlations"] = data[numeric].corr().round(4).to_dict()
    _save_plot("numerical_correlation", lambda: sns.heatmap(data[numeric].corr(), annot=True, cmap="coolwarm"))


def _run_text_analysis(data: pd.DataFrame, summary: Dict[str, Any]) -> None:
    summary["title_counts"] = data["Title"].value_counts().to_dict()
    summary["ticket_prefix_counts"] = data["TicketPrefix"].value_counts().head(20).to_dict()
    summary["deck_counts"] = data["Deck"].value_counts().to_dict()
    _count_plot(data, "Title", "title_distribution")
    _count_plot(data, "TicketPrefix", "ticket_prefix_distribution")
    _count_plot(data, "Deck", "deck_distribution")


def _run_train_test_comparison(train: pd.DataFrame, test: pd.DataFrame, summary: Dict[str, Any]) -> None:
    comparisons: Dict[str, Any] = {}
    for column in ["Age", "Fare", "Sex", "Pclass", "Embarked"]:
        comparisons[column] = {
            "train": train[column].value_counts(normalize=True, dropna=False).round(4).to_dict()
            if column in ["Sex", "Pclass", "Embarked"]
            else train[column].describe().round(4).to_dict(),
            "test": test[column].value_counts(normalize=True, dropna=False).round(4).to_dict()
            if column in ["Sex", "Pclass", "Embarked"]
            else test[column].describe().round(4).to_dict(),
        }
        combined = pd.concat(
            [train[[column]].assign(dataset="Train"), test[[column]].assign(dataset="Test")],
            ignore_index=True,
        )
        combined = combined.reset_index(drop=True)
        _save_plot(
            f"train_test_{column.lower()}",
            lambda combined=combined, column=column: (
                sns.histplot(data=combined, x=column, hue="dataset", stat="density", common_norm=False)
                if column in ["Age", "Fare"]
                else sns.countplot(data=combined, x=column, hue="dataset")
            ),
        )
    summary["train_test_distributions"] = comparisons


def _run_interactions(data: pd.DataFrame) -> None:
    _save_plot("interaction_sex_pclass", lambda: sns.barplot(data=data, x="Pclass", y=TARGET_COLUMN, hue="Sex", errorbar=None))
    _save_plot("interaction_age_sex", lambda: sns.boxplot(data=data, x="Sex", y="Age", hue=TARGET_COLUMN))
    _save_plot("interaction_fare_pclass", lambda: sns.boxplot(data=data, x="Pclass", y="Fare"))


def _recommendations() -> list[str]:
    return [
        "Title",
        "Deck",
        "Family_Size",
        "Is_Alone",
        "Fare_per_Person",
        "Ticket_Count",
        "Is_Mother",
    ]


def run_eda() -> Dict[str, Any]:
    """Run the complete EDA workflow and save figures and a JSON summary."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading Titanic train and test data")
    train, test = _load_data()
    original_train, original_test = train.copy(deep=True), test.copy(deep=True)
    LOGGER.info("Train shape: %s; test shape: %s", train.shape, test.shape)
    LOGGER.info("Train columns: %s", list(train.columns))
    LOGGER.info("Train dtypes:\n%s", train.dtypes)
    LOGGER.info("Train statistics:\n%s", train.describe(include="all"))
    LOGGER.info("Missing values:\n%s", _missing_summary(train))
    LOGGER.info("Duplicate rows: train=%d, test=%d", train.duplicated().sum(), test.duplicated().sum())
    print(f"Train shape: {train.shape}; test shape: {test.shape}")
    print(f"Columns: {list(train.columns)}")
    print(f"Data types:\n{train.dtypes}")
    print(f"Basic statistics:\n{train.describe(include='all')}")
    print(f"Missing values:\n{_missing_summary(train)}")
    print(f"Duplicate rows: train={train.duplicated().sum()}, test={test.duplicated().sum()}")

    data = _prepare_analysis_data(train)
    summary: Dict[str, Any] = {
        "train_shape": list(original_train.shape),
        "test_shape": list(original_test.shape),
        "columns": list(original_train.columns),
        "dtypes": {key: str(value) for key, value in original_train.dtypes.items()},
        "missing_values": _missing_summary(original_train).to_dict(orient="index"),
        "duplicate_rows": {"train": int(original_train.duplicated().sum()), "test": int(original_test.duplicated().sum())},
    }
    _run_univariate(data, summary)
    _run_bivariate(data, summary)
    _run_text_analysis(data, summary)
    _run_train_test_comparison(original_train, original_test, summary)
    _run_interactions(data)
    summary["candidate_features"] = _recommendations()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOGGER.info("EDA complete; summary saved to %s", SUMMARY_PATH)
    LOGGER.info("Candidate engineering features: %s", summary["candidate_features"])
    return summary


if __name__ == "__main__":
    run_eda()
