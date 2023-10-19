import mlflow
from dagshub import DAGsHubLogger


class BaseLogger:
    def __init__(self):
        self.logger = DAGsHubLogger()

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
        print("Loging metrics...")
        self.logger.log_metrics(metrics)

    def log_params(self, params: dict):
        mlflow.log_params(params)
        print("Loging params...")
        self.logger.log_hyperparams(params)
