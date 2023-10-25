import pandas as pd
import numpy as np
import hydra
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import yaml
from omegaconf import DictConfig
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from training.process import get_features, process_null, process_outliers, process_duplicate, process_categorical

# Define a test suite with specific steps
@test_steps("process_null", "process_duplicate", "get_features_step", "process_duplicate", "process_outliers", "process_categorical")
def test_processs_suite(test_step, steps_data):
    """
    Test suite to execute data processing steps.

    Args:
        test_step (str): The name of the test step.
        steps_data: Data container for sharing data between test steps.
    """
    if test_step == "process_null":
        process_null_step(steps_data)
    elif test_step == "process_duplicate":
        process_duplicate_step(steps_data)
    elif test_step == "get_features_step":
        get_features_step(steps_data)
    elif test_step == "process_outliers":
        process_outliers_step(steps_data)
    elif test_step == "process_categorical":
        process_categorical_step(steps_data)

def process_null_step(steps_data):
    """
    Process null values in the data.

    Args:
        steps_data: Data container for sharing data between test steps.
    """
    data = pd.DataFrame(
        {
            "age": [0, 50, 130, 130, 0, 50],
            "tstage": [1.0, 2.0, 4.0, 4.0, 1.0, 2.0],
            "nodalstatus": [0.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            "grade": [1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
            "erihc": ["No", "No", "Yes", "Yes", "No", "No"],
            "prihc": ["No", "No", "Yes", "Yes", "Yes", "Yes"],
            "pcr": [0, 1, 1, 1, 0, 1],
        }
    )

    schema = DataFrameSchema(
        {
            "age": Column(int, Check.in_range(0, 150)),
            "tstage": Column(float, Check.in_range(1.0, 4.0)),
            "nodalstatus": Column(float, Check.in_range(0.0, 2.0)),
            "grade": Column(float, Check.in_range(1.0, 3.0)),
            "erihc": Column(object, Check.isin(["No", "Yes"])),
            "prihc": Column(object, Check.isin(["No", "Yes"])),
            "pcr": Column(int, Check.isin([0, 1])),
        }
    )

    data = process_null(data)
    schema.validate(data)
    steps_data.data = data

def process_duplicate_step(steps_data):
    """
    Remove duplicate rows from the data.

    Args:
        steps_data: Data container for sharing data between test steps.
    """
    # Check for duplicate rows using the duplicated() method
    steps_data.data = process_duplicate(steps_data.data)
    duplicate_rows = steps_data.data[steps_data.data.duplicated()]
    # Check if there are duplicate rows based on the schema
    assert duplicate_rows.empty, "Duplicate rows found based on the schema."

def get_features_step(steps_data):
    """
    Extract features and target variable from the data.

    Args:
        steps_data: Data container for sharing data between test steps.
    """
    features = [
        "age",
        "tstage",
        "nodalstatus",
        "grade",
        "erihc",
        "prihc",
    ]
    target = "pcr"
    y, X = get_features(target, features, steps_data.data)

    schema = DataFrameSchema(
        {
            "age": Column(int, Check.in_range(0, 150)),
            "tstage": Column(float, Check.in_range(1.0, 4.0)),
            "nodalstatus": Column(float, Check.in_range(0.0, 2.0)),
            "grade": Column(float, Check.in_range(1.0, 3.0)),
            "erihc": Column(object, Check.isin(["No", "Yes"])),
            "prihc": Column(object, Check.isin(["No", "Yes"])),
            "pcr": Column(int, Check.isin([0, 1])),
        }
    )
    schema.validate(pd.concat([X, y], axis=1))
    steps_data.data = X

def process_outliers_step(steps_data):
    """
    Check and process outliers in the data.

    Args:
        steps_data: Data container for sharing data between test steps.
    """
    # Initialize Hydra
    hydra.initialize(config_path="../config", job_name="main", version_base="1.1")

    # Load the Hydra configuration
    config = hydra.compose(config_name="main")

    validation_criteria = {
        "age": (0, 120),
        "tstage": (1.0, 3.0),
        "nodalstatus": (0.0, 2.0),
        "grade": (1.0, 3.0),
        "erihc": ["No", "Yes"],
        "prihc": ["No", "Yes"],
    }

    data = process_outliers(steps_data.data, config.process.features_range)

    for column, criterion in validation_criteria.items():
        if isinstance(criterion, tuple):
            # Check if column values are within the specified range
            assert (data[column] >= criterion[0]).all() and (data[column] <= criterion[1]).all()
        elif isinstance(criterion, list):
            # Check if column values are in the specified list
            assert data[column].isin(criterion).all()

    GlobalHydra.instance().clear()
    steps_data.data = data

def process_categorical_step(steps_data):
    """
    Process categorical data in the dataframe.

    Args:
        steps_data: Data container for sharing data between test steps.
    """
    data = pd.DataFrame(
        {
            "age": [0, 50, 130, 130, 0, 50],
            "tstage": [1.0, 2.0, 4.0, 4.0, 1.0, 2.0],
            "nodalstatus": [0.0, 1.0, 2.0, 2.0, 0.0, 1.0],
            "grade": [1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
            "erihc": ["No", "No", "Yes", "Yes", "No", "No"],
            "prihc": ["No", "No", "Yes", "Yes", "Yes", "Yes"],
            "pcr": [0, 1, 1, 1, 0, 1],
        }
    )

    schema = DataFrameSchema(
        {
            "age": Column(int, Check.in_range(0, 150)),
            "tstage": Column(float, Check.in_range(1.0, 4.0)),
            "nodalstatus": Column(float, Check.in_range(0.0, 2.0)),
            "grade": Column(float, Check.in_range(1.0, 3.0)),
            "erihc_No": Column(float, Check.isin([0.0, 1.0])),
            "erihc_Yes": Column(float, Check.isin([0.0, 1.0])),
            "prihc_No": Column(float, Check.isin([0.0, 1.0])),
            "prihc_Yes": Column(float, Check.isin([0.0, 1.0])),
        }
    )

    categorical_columns = data.select_dtypes(include=['category', 'object']).columns.tolist()
    data = process_categorical(data, categorical_columns)
    schema.validate(data)
