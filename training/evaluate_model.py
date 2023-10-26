import warnings

warnings.filterwarnings(action="ignore")

import hydra
import joblib
import mlflow
import pandas as pd
from helper import BaseLogger
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import os


# os.environ['MLFLOW_TRACKING_USERNAME'] = 'GitHubAlejandroDR'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = '99f73425f37db3558a5a5f508d47e397f8348e04'
# mlflow_tracking_ui: https://dagshub.com/GitHubAlejandroDR/cancer-clinical-test.mlflow
logger = BaseLogger()


def load_data(path: DictConfig):
    """
    Load test data from CSV files.

    Args:
        path (DictConfig): Configuration specifying file paths.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the test features and labels.
    """
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_test, y_test

def load_model(model_path: str):
    """
    Load a machine learning model from a file.

    Args:
        model_path (str): The path to the saved model.

    Returns:
        XGBClassifier: The loaded XGBoost classifier model.
    """
    return joblib.load(model_path)

def predict(model: XGBClassifier, X_test: pd.DataFrame):
    """
    Make predictions using a trained XGBoost model.

    Args:
        model (XGBClassifier): The trained XGBoost classifier.
        X_test (pd.DataFrame): The test data for making predictions.

    Returns:
        pd.Series: Predicted labels.
    """
    return model.predict(X_test)

def log_params(model: XGBClassifier, features: list):
    """
    Log model parameters and feature information.

    Args:
        model (XGBClassifier): The trained XGBoost classifier.
        features (list): List of features used in the model.

    Returns:
        None
    """
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()
    for arg, value in model_params.items():
        logger.log_params({arg: value})
    logger.log_params({"features": features})

def log_metrics(**metrics: dict):
    """
    Log evaluation metrics.

    Args:
        **metrics (dict): Named evaluation metrics and their values.

    Returns:
        None
    """
    logger.log_metrics(metrics)

@hydra.main(version_base=None, config_path="../config", config_name="main")
def evaluate(config: DictConfig):
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    mlflow.set_experiment("clinical-cancer")

    with mlflow.start_run():
        # Load data and model
        X_test, y_test = load_data(config.processed)
        model = load_model(abspath(config.model.path))

        # Get predictions
        prediction = predict(model, X_test)

        # Get metrics
        f1 = f1_score(y_test, prediction)
        print(f"F1 Score of this model is {f1}.")

        accuracy = accuracy_score(y_test, prediction)
        print(f"Accuracy Score of this model is {accuracy}.")

        # Log metrics
        log_params(model, config.process.features)
        log_metrics(f1_score=f1, accuracy_score=accuracy)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    evaluate()
