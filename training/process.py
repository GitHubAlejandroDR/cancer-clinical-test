import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""
This script processes raw data according to the provided configuration.
"""


def get_data(raw_path: str, sep: str):
    """
    Load data from a CSV file.

    Args:
        raw_path (str): Path to the raw data file.
        sep (str): Delimiter used in the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(raw_path, sep=sep)
    return data


def process_null(data: pd.DataFrame):
    """
    Remove rows with null or missing values from the dataset.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Data with null rows removed.
    """
    return data.dropna(axis=0)


def process_duplicate(data: pd.DataFrame):
    """
    Remove duplicate rows from the dataset.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Data with duplicate rows removed.
    """
    return data.drop_duplicates(keep="first")


def get_features(target: str, features: list, data: pd.DataFrame):
    """
    Extract target and selected features from the dataset.

    Args:
        target (str): Name of the target variable.
        features (list): List of selected feature names.
        data (pd.DataFrame): Input data.

    Returns:
        pd.Series: Target variable (y)
        pd.DataFrame: Selected features (X)
    """
    y, X = data[target], data[features]
    return y, X


def process_outliers(data: pd.DataFrame, features_range: DictConfig):
    """
    Remove outliers from the dataset based on specified feature ranges.

    Args:
        data (pd.DataFrame): Input data.
        features_range (DictConfig): Configuration for feature range checks.

    Returns:
        pd.DataFrame: Data with outliers removed.
    """
    return data[
        (data[features_range.name] <= features_range.max)
        & (data[features_range.name] >= features_range.min)
    ]


def process_dtypes(X: pd.DataFrame):
    """
    Process data types if needed (e.g., data type conversions).

    Args:
        X (pd.DataFrame): Input features data.

    Returns:
        pd.DataFrame: Processed feature data.
    """
    return X


def process_categorical(X: pd.DataFrame, categorical_features: list):
    """
    Encode categorical features using one-hot encoding.

    Args:
        X (pd.DataFrame): Input features data.
        categorical_features (list): List of categorical feature names.

    Returns:
        pd.DataFrame: Data with one-hot encoded categorical features.
    """
    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    categorical_encoded = encoder.fit_transform(X[categorical_features])
    X = pd.concat(
        [X.iloc[:, ~X.columns.isin(categorical_features)], categorical_encoded], axis=1
    )
    return X


@hydra.main(version_base=None, config_path="../config", config_name="main")
def process_data(config: DictConfig):
    """
    Process the raw data as per the configuration provided.

    Args:
        config (DictConfig): Configuration parameters.

    Returns:
        None
    """
    data = get_data(abspath(config.raw.path), config.process.sep)

    data = process_null(data)

    data = process_duplicate(data)

    data = process_outliers(data, config.process.features_range)

    print(data)

    y, X = get_features(config.process.target, config.process.features, data)

    X = process_categorical(X, config.process.categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # Save data
    X_train.to_csv(abspath(config.processed.X_train.path), index=False)
    X_test.to_csv(abspath(config.processed.X_test.path), index=False)
    y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    y_test.to_csv(abspath(config.processed.y_test.path), index=False)


if __name__ == "__main__":
    process_data()
