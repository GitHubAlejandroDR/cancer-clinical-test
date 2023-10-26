import mlflow
from dagshub import DAGsHubLogger

class BaseLogger:
    def __init__(self):
        """
        Initialize a logging object for metrics and parameters.

        This class provides a way to log metrics and parameters using both MLflow
        and DAGsHubLogger.

        Returns:
            None
        """
        self.logger = DAGsHubLogger()

    def log_metrics(self, metrics: dict):
        """
        Log metrics using MLflow and DAGsHubLogger.

        Args:
            metrics (dict): A dictionary of named metrics and their values.

        Returns:
            None
        """
        mlflow.log_metrics(metrics)
        print("Logging metrics...")
        self.logger.log_metrics(metrics)

    def log_params(self, params: dict):
        """
        Log parameters using MLflow and DAGsHubLogger.

        Args:
            params (dict): A dictionary of named parameters and their values.

        Returns:
            None
        """
        mlflow.log_params(params)
        print("Logging parameters...")
        self.logger.log_hyperparams(params)
