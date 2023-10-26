import warnings

warnings.filterwarnings(action="ignore")

from functools import partial
from typing import Callable

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

"""
This script trains an XGBoost model using hyperparameter optimization.
"""

def load_data(path: DictConfig):
    """
    Load training and testing data.

    Parameters:
        path (DictConfig): Configuration path containing file paths.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Training and testing data for features and labels.
    """
    X_train = pd.read_csv(abspath(path.X_train.path))
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_train = pd.read_csv(abspath(path.y_train.path))
    y_test = pd.read_csv(abspath(path.y_test.path))
    return X_train, X_test, y_train, y_test

def get_objective(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: DictConfig,
    space: dict,
):
    """
    Define the optimization objective.

    Parameters:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.DataFrame): Training data labels.
        X_test (pd.DataFrame): Testing data features.
        y_test (pd.DataFrame): Testing data labels.
        config (DictConfig): Configuration object.
        space (dict): Hyperparameter search space.

    Returns:
        dict: Dictionary with 'loss', 'status', and the best model.
    """
    model = XGBClassifier(
        use_label_encoder=config.model.use_label_encoder,
        objective=config.model.objective,
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    model.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        eval_metric=config.model.eval_metric,
        early_stopping_rounds=config.model.early_stopping_rounds,
    )
    prediction = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, prediction)
    print("SCORE:", accuracy)
    return {"loss": -accuracy, "status": STATUS_OK, "model": model}

def optimize(objective: Callable, space: dict):
    """
    Perform hyperparameter optimization.

    Parameters:
        objective (Callable): The optimization objective function.
        space (dict): Hyperparameter search space.

    Returns:
        XGBClassifier: The best trained XGBoost model.
    """
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)
    best_model = trials.results[
        np.argmin([r["loss"] for r in trials.results])
    ]["model"]
    return best_model

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def train(config: DictConfig):
    """Train an XGBoost model with hyperparameter optimization."""
    
    X_train, X_test, y_train, y_test = load_data(config.processed)

    # Define space
    space = {
        "max_depth": hp.quniform("max_depth", **config.model.max_depth),
        "gamma": hp.uniform("gamma", **config.model.gamma),
        "reg_alpha": hp.quniform("reg_alpha", **config.model.reg_alpha),
        "reg_lambda": hp.uniform("reg_lambda", **config.model.reg_lambda),
        "colsample_bytree": hp.uniform(
            "colsample_bytree", **config.model.colsample_bytree
        ),
        "min_child_weight": hp.quniform(
            "min_child_weight", **config.model.min_child_weight
        ),
        "n_estimators": config.model.n_estimators,
        "seed": config.model.seed,
    }
    objective = partial(
        get_objective, X_train, y_train, X_test, y_test, config
    )

    # Find best model
    best_model = optimize(objective, space)

    # Save model
    joblib.dump(best_model, abspath(config.model.path))

if __name__ == "__main__":
    train()
