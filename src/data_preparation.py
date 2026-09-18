"""
data_preparation.py

Reusable functions for loading, cleaning, and splitting the O-Level Math
score dataset. Logic ported from eda.ipynb.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


# ---------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - CCA: missing values assumed to mean no CCA participation -> "NONE"
    - final_test: rows with missing target are dropped (can't train on them)
    - attendance_rate: missing values filled with the median
    """
    df = df.copy()

    df["CCA"] = df["CCA"].fillna("NONE")
    df = df.dropna(subset=["final_test"])

    median_attendance = df["attendance_rate"].median()
    df["attendance_rate"] = df["attendance_rate"].fillna(median_attendance)

    return df


# ---------------------------------------------------------------------
# Invalid / negative values
# ---------------------------------------------------------------------
def handle_invalid_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Negative ages and implausible ages (< 15) for O-level students are
    treated as invalid and imputed with the median age.
    """
    df = df.copy()

    median_age = round(df["age"].median())

    df.loc[df["age"] < 0, "age"] = median_age
    df.loc[df["age"] < 15, "age"] = median_age

    return df


# ---------------------------------------------------------------------
# Standardising categorical values
# ---------------------------------------------------------------------
def standardize_tuition(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise 'Y'/'N' tuition values to 'Yes'/'No'."""
    df = df.copy()
    df["tuition"] = df["tuition"].replace({"Y": "Yes", "N": "No"})
    return df


def standardize_categorical_text(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """Strip whitespace and title-case categorical text columns."""
    df = df.copy()
    for column in categorical_features:
        df[column] = df[column].astype(str).str.strip().str.title()
    return df


# ---------------------------------------------------------------------
# Dropping unnecessary columns
# ---------------------------------------------------------------------
def drop_unnecessary_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Drop columns that provide no predictive value (default: id, bag_color)."""
    if columns is None:
        columns = ["student_id", "bag_color"]
    df = df.copy()
    df.drop(columns=columns, inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------
def clean_data(df: pd.DataFrame, categorical_features: list = None) -> pd.DataFrame:
    """Run the full cleaning pipeline in the same order as the EDA notebook."""
    if categorical_features is None:
        categorical_features = [
            "direct_admission",
            "CCA",
            "learning_style",
            "gender",
            "tuition",
            "sleep_time",
            "wake_time",
            "mode_of_transport",
        ]

    df = handle_missing_values(df)
    df = handle_invalid_age(df)
    df = standardize_tuition(df)
    df = drop_unnecessary_columns(df)
    df = standardize_categorical_text(df, categorical_features)

    return df


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target: str = "final_test",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split cleaned data into train/test feature and target sets."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------
def prepare_data(filepath: str, target: str = "final_test", test_size: float = 0.2, random_state: int = 42):
    """Load, clean, and split the data in one call."""
    df = load_data(filepath)
    df = clean_data(df)
    return split_data(df, target=target, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data("/home/samuel/aiap/AIML_foundation/AI-ML-foundation-regression-/data/regression_bonus_practice_data.csv")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)