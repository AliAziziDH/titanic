import os
import pandas as pd
import pytest

def test_train_clean_shape():
    """Verify data/processed/train_clean.csv has exactly 891 rows."""
    file_path = "data/processed/train_clean.csv"
    assert os.path.exists(file_path), f"File missing: {file_path}. Data pipeline needs to be run first."
    df = pd.read_csv(file_path)
    assert df.shape[0] == 891, f"Expected 891 rows in {file_path}, but got {df.shape[0]}."

def test_test_clean_shape():
    """Verify data/processed/test_clean.csv has exactly 418 rows."""
    file_path = "data/processed/test_clean.csv"
    assert os.path.exists(file_path), f"File missing: {file_path}. Data pipeline needs to be run first."
    df = pd.read_csv(file_path)
    assert df.shape[0] == 418, f"Expected 418 rows in {file_path}, but got {df.shape[0]}."

def test_submission_stacking_shape():
    """Verify submissions/submission_stacking.csv (if present) has exactly 418 rows, correct columns and IDs."""
    file_path = "submissions/submission_stacking.csv"
    if not os.path.exists(file_path):
        pytest.skip(f"Submission file {file_path} not found. Skipping submission validation.")

    df = pd.read_csv(file_path)

    # Check rows
    assert df.shape[0] == 418, f"Expected 418 rows in {file_path}, but got {df.shape[0]}."

    # Check columns
    expected_cols = ["PassengerId", "Survived"]
    assert list(df.columns) == expected_cols, f"Expected columns {expected_cols}, but got {list(df.columns)}."

    # Check PassengerId range (892 to 1309)
    assert df["PassengerId"].min() == 892, f"Expected minimum PassengerId 892, but got {df['PassengerId'].min()}."
    assert df["PassengerId"].max() == 1309, f"Expected maximum PassengerId 1309, but got {df['PassengerId'].max()}."

    # Check that Survived contains only 0 and 1
    assert set(df["Survived"].unique()).issubset({0, 1}), f"Expected Survived to be binary (0, 1), but got {df['Survived'].unique()}"
