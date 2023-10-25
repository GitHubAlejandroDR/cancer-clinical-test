import joblib
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestPerformance
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath
from training.train_model import load_data


"""
    Test the performance of a trained XGBoost model on a test dataset.

    The function uses Deepchecks to assess the accuracy and F1-score of the model on the test dataset.
    It also checks the relative degradation between the train and test datasets. 

    Returns:
        None
    """

def test_xgboost():
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="main")

    model_path = abspath(config.model.path)
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = load_data(config.processed)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_ds = Dataset(train_df, label="pcr")
    validation_ds = Dataset(test_df, label="pcr")

    check = TrainTestPerformance(scorers=["f1"])
    check.add_condition_test_performance_greater_than(0.9)
    check.add_condition_train_test_relative_degradation_less_than(0.3)
    check.run(train_ds, validation_ds, model)

