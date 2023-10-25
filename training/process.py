import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""_summary_
"""

def get_data(raw_path: str, sep: str):
    data = pd.read_csv(raw_path, sep=sep)
    return data

# Null/Na Values
def process_null (data: pd.DataFrame):
    """_summary_

    Arguments:
        data -- _description_

    Returns:
        _description_
    """
    return data.dropna(axis=0)

# Duplicates
def process_duplicate (data: pd.DataFrame):
    return data.drop_duplicates(keep="first")

def get_features(target: str, features: list, data: pd.DataFrame):
    y, X = data[target], data[features]
    return y, X

# Outliers
def process_outliers (data: pd.DataFrame, features_range: DictConfig):
    return data[(data[features_range.name] <= features_range.max) & (data[features_range.name] >= features_range.min)]

# Data types
def process_dtypes (X: pd.DataFrame):
    
    return X

# Encoding
def process_categorical (X: pd.DataFrame, categorical_features: list):
    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    categorical_encoded = encoder.fit_transform(X[categorical_features])
    X = pd.concat([X.iloc[:,~X.columns.isin(categorical_features)], categorical_encoded], axis=1)
    return X

# Columns confusion

# Feature selection

# Scaling Normalization/Standardization

# Dimensionality reduction

# Bining and discretization



@hydra.main(version_base=None, config_path="../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    
    
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
    # X_train.to_csv(abspath(config.processed.X_train.path), index=False)
    # X_test.to_csv(abspath(config.processed.X_test.path), index=False)
    # y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    # y_test.to_csv(abspath(config.processed.y_test.path), index=False)

"""_summary_
"""

if __name__ == "__main__":
    process_data()
