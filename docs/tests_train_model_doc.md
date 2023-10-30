# XGBoost Model Testing Documentation

## Overview
This documentation explains the testing process for a trained XGBoost model using Deepchecks. The code tests the model's accuracy, F1-score, and relative degradation between train and test datasets, ensuring that they meet specific criteria.

## Code Description
The provided Python code tests a trained XGBoost model by performing the following tasks:

1. Load the model and dataset using Hydra configuration.
2. Use Deepchecks to assess the model's accuracy and F1-score on the test dataset.
3. Ensure that both accuracy and F1-score are greater than 0.9.
4. Check the relative degradation between train and test datasets, ensuring it's less than 0.3.
5. Output the ROC AUC score for reference.

## Prerequisites
Before running the code, make sure you have the following installed:

- Python
- Deepchecks
- XGBoost
- Hydra
- Joblib
- Pandas

## Inputs
- A trained XGBoost model.
- Test dataset for the XGBoost model.

## Outputs
- Pass or fail status of the test.
- Printed ROC AUC score on the test set.

## Usage

```python
# To run the entire test suite
$ pytest /path_to_test_folder/
