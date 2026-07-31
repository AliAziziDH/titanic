"""Feature engineering utilities for the Titanic project."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, KAGGLE_INPUT_DIR, TARGET_COLUMN


LOGGER = logging.getLogger("titanic.features")

TITLE_MAPPING = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
DECK_MAPPING = {"U": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
ESSENTIAL_COLUMNS = [
    "PassengerId",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


def _extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and numerically encode passenger titles."""
    titles = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).fillna("Rare")
    titles = titles.where(titles.isin(TITLE_MAPPING), "Rare")
    df["Title"] = titles
    df["Title_Encoded"] = titles.map(TITLE_MAPPING).astype("int8")
    return df


def _extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cabin deck, presence, and numeric deck encoding."""
    cabin = df["Cabin"].fillna("").astype(str)
    decks = cabin.str.extract(r"^([A-Za-z])", expand=False).str.upper().fillna("U")
    decks = decks.where(decks.isin(DECK_MAPPING), "U")
    df["Deck"] = decks
    df["Deck_Encoded"] = decks.map(DECK_MAPPING).astype("int8")
    df["Has_Cabin"] = df["Cabin"].notna().astype("int8")
    return df


def _extract_family_name(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the surname portion of each passenger name."""
    df["Family_Name"] = df["Name"].str.split(",", n=1).str[0].str.strip()
    return df


def _extract_ticket_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract alphabetic and punctuation ticket prefixes."""
    tickets = df["Ticket"].fillna("").astype(str).str.strip()
    prefixes = tickets.str.extract(r"^([A-Za-z/.]+)", expand=False)
    df["Ticket_Prefix"] = prefixes.fillna("NUMERIC").str.replace(".", "", regex=False).str.upper()
    return df


def _create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create family size and family-size category features."""
    df["Family_Size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["Is_Alone"] = (df["Family_Size"] == 1).astype("int8")
    df["Family_Size_Category"] = pd.cut(
        df["Family_Size"],
        bins=[0, 1, 4, np.inf],
        labels=["Alone", "Small", "Large"],
        include_lowest=True,
    )
    return df


def _create_ticket_features(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create passenger count and group indicators for each ticket."""
    ticket_counts = (reference if reference is not None else df)["Ticket"].value_counts(dropna=False)
    df["Ticket_Count"] = df["Ticket"].map(ticket_counts)
    df["Ticket_Count"] = df["Ticket_Count"].fillna(1).astype("int64")
    df["Is_Group"] = (df["Ticket_Count"] > 1).astype("int8")
    return df


def _create_fare_features(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create per-person fare and quantile-based fare category."""
    family_size = df["Family_Size"].replace(0, 1)
    df["Fare_per_Person"] = df["Fare"] / family_size
    try:
        reference_fare = (reference if reference is not None else df)["Fare"].dropna()
        quantiles = reference_fare.quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy()
        edges = np.unique(quantiles)
        labels = ["Low", "Medium", "High", "Very High"][: len(edges) - 1]
        df["Fare_Bin"] = pd.cut(df["Fare"], bins=edges, labels=labels, include_lowest=True)
    except ValueError:
        df["Fare_Bin"] = pd.Series(pd.NA, index=df.index, dtype="string")
    return df


def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interactions between sex, class, title, and sex."""
    df["Sex_Pclass"] = df["Sex"].astype("string") + "_" + df["Pclass"].astype("string")
    df["Title_Sex"] = df["Title"].astype("string") + "_" + df["Sex"].astype("string")
    return df


def _create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create mother, age band, and grouped deck features."""
    df["Is_Mother"] = (
        df["Sex"].eq("female") & df["Title"].eq("Mrs") & df["Parch"].fillna(0).gt(0)
    ).astype("int8")
    age = df["Age"].fillna(-1)
    df["Age_Band"] = pd.cut(
        age,
        bins=[-np.inf, -0.5, 12, 18, 49.999999, np.inf],
        labels=["Missing", "Child", "Teen", "Adult", "Senior"],
    )
    df["Deck_Group"] = df["Deck"].map(
        lambda deck: "High" if deck in {"A", "B", "C"} else "Low" if deck in {"D", "E", "F", "G"} else "U"
    )
    return df


def engineer_features(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a new DataFrame with Titanic feature engineering applied."""
    required = set(ESSENTIAL_COLUMNS + ["Name", "Ticket", "Cabin"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    LOGGER.info("Engineering features for %d rows", len(df))
    engineered = df.copy(deep=True)
    _extract_title(engineered)
    _extract_family_name(engineered)
    _extract_deck(engineered)
    _extract_ticket_prefix(engineered)
    _create_family_features(engineered)
    _create_ticket_features(engineered, reference_df)
    _create_fare_features(engineered, reference_df)
    _create_interaction_features(engineered)
    _create_advanced_features(engineered)

    columns = ESSENTIAL_COLUMNS + [
        "Title",
        "Title_Encoded",
        "Family_Name",
        "Deck",
        "Deck_Encoded",
        "Has_Cabin",
        "Deck_Group",
        "Family_Size",
        "Is_Alone",
        "Family_Size_Category",
        "Ticket_Count",
        "Is_Group",
        "Ticket_Prefix",
        "Fare_per_Person",
        "Fare_Bin",
        "Sex_Pclass",
        "Title_Sex",
        "Is_Mother",
        "Age_Band",
    ]
    if TARGET_COLUMN in engineered.columns:
        columns.insert(1, TARGET_COLUMN)
    result = engineered.loc[:, columns].copy()
    LOGGER.info("Created %d engineered columns", len(result.columns))
    return result


def save_engineered_data(
    train: pd.DataFrame, test: pd.DataFrame, output_dir: Path = DATA_PROCESSED_DIR
) -> Tuple[Path, Path]:
    """Engineer and save train/test datasets to the processed-data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_engineered.csv"
    test_path = output_dir / "test_engineered.csv"
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    engineered = engineer_features(combined, reference_df=combined)
    engineer_train = engineered.iloc[: len(train)].copy()
    engineer_test = engineered.iloc[len(train):].copy()
    engineer_train.to_csv(train_path, index=False)
    engineer_test.drop(columns=[TARGET_COLUMN], errors="ignore").to_csv(test_path, index=False)
    LOGGER.info("Saved engineered data to %s and %s", train_path, test_path)
    return train_path, test_path


if __name__ == "__main__":
    from src.config import get_input_dir

    input_dir = get_input_dir()
    train_data = pd.read_csv(input_dir / "train.csv")
    test_data = pd.read_csv(input_dir / "test.csv")
    save_engineered_data(train_data, test_data)
