"""Prepare Titanic data for machine learning.

Run: python titanic_pipeline.py --csv Titanic-Dataset.csv
Leave out --csv to load the Seaborn dataset.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = {
    "survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"
}
NUMERIC_COLUMNS = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_COLUMNS = ["sex", "class", "embarked"]


class DataValidationError(ValueError):
    """Something is missing or incorrect in the input data."""


def load_data(csv_path=None):
    """Read the CSV or load Titanic from Seaborn."""
    if csv_path is None:
        import seaborn as sns

        data = sns.load_dataset(
            "titanic", cache=False, data_home=tempfile.gettempdir()
        )
    else:
        data = pd.read_csv(csv_path)
    # This makes "Age" and "age" work the same way.
    data.columns = data.columns.str.strip().str.lower()
    return data


def validate_data(data):
    """Check the required columns and values."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")
    if data.empty or data.columns.duplicated().any():
        raise DataValidationError("The dataset is empty or has duplicate columns.")
    if data["survived"].isna().any() or not data["survived"].isin([0, 1]).all():
        raise DataValidationError("Survived must contain only 0 and 1, without blanks.")
    if set(data["survived"].unique()) != {0, 1}:
        raise DataValidationError("Both survival classes (0 and 1) are required.")
    if data["survived"].value_counts().min() < 2:
        raise DataValidationError("Each survival class needs at least two rows.")
    for column in ["pclass"] + NUMERIC_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.dropna().astype(float)).any():
            raise DataValidationError(f"Invalid numeric values in {column}.")
        if (numeric.dropna() < 0).any():
            raise DataValidationError(f"Negative values are not allowed in {column}.")
    pclass = pd.to_numeric(data["pclass"], errors="coerce")
    if not pclass.dropna().isin([1, 2, 3]).all():
        raise DataValidationError("Pclass must be 1, 2 or 3 when present.")


def clean_text(data):
    """Remove extra spaces and use lowercase text."""
    data = data.copy()
    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].astype("string").str.strip().str.lower()
            data[column] = data[column].replace("", pd.NA)
    for column in ["pclass"] + NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    # The CSV has Pclass; Seaborn also has class. Use one consistent version.
    data["class"] = data["pclass"].map({1: "first", 2: "second", 3: "third"})
    for column, allowed in {
        "sex": {"female", "male"}, "embarked": {"c", "q", "s"}
    }.items():
        unexpected = set(data[column].dropna().unique()) - allowed
        if unexpected:
            raise DataValidationError(f"Unexpected {column} values: {unexpected}")
    return data


def drop_useless(data):
    """Remove columns we do not need for this project."""
    # "alive" gives away the answer, while cabin and deck have many blanks.
    # We also leave out identifiers and columns that repeat other information.
    columns_to_drop = [
        "alive", "deck", "cabin", "name", "ticket", "passengerid",
        "pclass", "embark_town", "who", "adult_male", "alone",
    ]
    data = data.drop(columns=columns_to_drop, errors="ignore")
    return data[["survived"] + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()


def handle_missing(X_train, X_test):
    """Fill missing numbers with medians and categories with modes."""
    X_train, X_test = X_train.copy(), X_test.copy()
    fill_values = {}
    for column in X_train.columns:
        observed = X_train[column].dropna()
        if observed.empty:
            raise DataValidationError(f"No training values available for {column}.")
        if column in NUMERIC_COLUMNS:
            fill_value = observed.median()
        else:
            fill_value = observed.mode().iloc[0]
        fill_values[column] = fill_value
        X_train[column] = X_train[column].fillna(fill_value)
        X_test[column] = X_test[column].fillna(fill_value)
    return X_train, X_test, fill_values


def encode_categoricals(X_train, X_test):
    """Turn sex, passenger class and boarding port into numbers."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for frame in (X_train, X_test):
        frame["sex"] = frame["sex"].map({"female": 0, "male": 1}).astype(int)
        frame["class"] = frame["class"].map(
            {"first": 1, "second": 2, "third": 3}
        ).astype(int)

    # Give each boarding port its own column, using the training ports.
    # A new port in the test set gets zeros in these columns.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_values = encoder.fit_transform(X_train[["embarked"]])
    test_values = encoder.transform(X_test[["embarked"]])
    names = encoder.get_feature_names_out(["embarked"])
    X_train = pd.concat([
        X_train.drop(columns="embarked"),
        pd.DataFrame(train_values, columns=names, index=X_train.index),
    ], axis=1)
    X_test = pd.concat([
        X_test.drop(columns="embarked"),
        pd.DataFrame(test_values, columns=names, index=X_test.index),
    ], axis=1)
    return X_train, X_test, encoder


def split_data(data):
    """Keep 80% for training and 20% for testing."""
    X = data.drop(columns="survived")
    y = data["survived"].astype(int)
    # stratify keeps a similar share of survivors in each set.
    # The fixed seed gives the same split each time we run the script.
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )


def scale_features(X_train, X_test):
    """Put age, family counts and fare on a similar scale."""
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])
    # Leave sex, class and port codes as they are.
    return X_train, X_test, scaler


def main():
    """Run the preparation steps and save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Optional local Titanic CSV")
    parser.add_argument("--output", type=Path, default=Path("prepared_data"))
    args = parser.parse_args()

    try:
        data = load_data(args.csv)
        validate_data(data)
        data = clean_text(data)
        data = drop_useless(data)

        # Split first: test rows should not affect how training data is filled
        # or scaled. This avoids data leakage.
        X_train, X_test, y_train, y_test = split_data(data)
        X_train, X_test, fill_values = handle_missing(X_train, X_test)
        X_train, X_test, encoder = encode_categoricals(X_train, X_test)
        X_train, X_test, scaler = scale_features(X_train, X_test)

        if not X_train.columns.equals(X_test.columns):
            raise DataValidationError("Train/test feature columns differ.")
        for frame in (X_train, X_test):
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise DataValidationError("Prepared features contain missing/infinite values.")

        args.output.mkdir(parents=True, exist_ok=True)
        # Keep features and labels in the same row order when saving.
        for name, frame in {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }.items():
            frame.to_csv(args.output / f"{name}.csv", index=False)

        summary = "\n".join([
            "TITANIC SURVIVOR DATA PREPARATION PIPELINE",
            f"Source: {args.csv if args.csv else 'Seaborn Titanic'}",
            f"Passengers: {len(data)}",
            f"Training features: {X_train.shape}",
            f"Testing features: {X_test.shape}",
            f"Survival rate (all/train/test): {data.survived.mean():.4f} / "
            f"{y_train.mean():.4f} / {y_test.mean():.4f}",
            f"Features: {', '.join(X_train.columns)}",
            "Missing values after preparation: 0",
            "Missing-value replacements and scaling use training data only.",
            "No prediction model was trained.",
        ])
        (args.output / "run_summary.txt").write_text(summary + "\n", encoding="utf-8")
        print(summary)
        print(f"\nSaved results to: {args.output.resolve()}")
        return 0
    except DataValidationError as error:
        print(f"Data validation failed: {error}", file=sys.stderr)
    except Exception as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        print("Check the input path, installed libraries and internet access for Seaborn.",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
