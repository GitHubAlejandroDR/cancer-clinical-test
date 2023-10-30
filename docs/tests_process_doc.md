# Test Suite Documentation

This README provides an overview of the test suite for data processing using Pandas, Pandera, and Hydra. The test suite includes several steps to process, clean, and validate the data.

## Test Steps

The test suite consists of the following test steps:

1. **Process Null Values (process_null)**
    - Description: This step processes null values in the dataset.
    - Input: Raw data with potential null values.
    - Output: Data without null values.
    
2. **Remove Duplicate Rows (process_duplicate)**
    - Description: This step removes duplicate rows from the dataset.
    - Input: Processed data (output of step 1).
    - Output: Data without duplicate rows.
    
3. **Extract Features and Target (get_features_step)**
    - Description: This step extracts features and the target variable from the data.
    - Input: Processed data (output of step 2).
    - Output: Features (X) and the target (y).

4. **Process Outliers (process_outliers)**
    - Description: Check and process outliers in the data.
    - Input: Processed data (output of step 3).
    - Output: Data with outliers processed or removed based on defined criteria.

5. **Process Categorical Data (process_categorical)**
    - Description: Process categorical data, including one-hot encoding.
    - Input: Processed data (output of step 4).
    - Output: Data with categorical data processed.

## Data and Configuration

### Data
Ensure that you have prepared your data as per the expected input of each test step. It's essential that the data adheres to the required format and structure for each test to run successfully.

### Configuration
Some test steps may require specific configuration parameters to define criteria for data processing. These configurations should be provided in a separate configuration file or as part of the testing framework setup.

## Dependencies

The test suite relies on the following Python libraries:

- Pandas: Data manipulation and cleaning.
- Pandera: Data validation using data schemas.
- Hydra: Configuration management for data processing criteria.
- pytest: Testing framework for executing the test steps.
- pytest-steps: Extension for structuring tests using steps.

Make sure to install these dependencies to run the tests successfully.

## Usage

```python
# To run the entire test suite
$ pytest /path_to_test_folder/

