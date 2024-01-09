import os
import logging
import pytest
import pandas as pd
import churn_library_solution as cls
from datetime import datetime
from constants import data_path, eda_image_folder
from exceptions import NonBinaryTargetException


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def import_data():
    return cls.import_data


@pytest.fixture(scope="module")
def perform_eda():
    return cls.perform_eda


@pytest.fixture(scope="module")
def perform_feature_engineering():
    return cls.perform_feature_engineering


def run_tests():
    logging.info("Starting tests at %s", datetime.now())


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    try:
        df = import_data(data_path)
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have "
                      "rows and columns")
        raise err

    try:
        assert df.isnull().sum().sum() == 0
    except AssertionError as err:
        logging.error("Testing import_data: The file has missing values")
        raise err

    logging.info("SUCCESS: data imported successfully")


def test_eda(perform_eda):
    """
    Test the perform_eda function.

    Parameters:
    perform_eda (function): The EDA function to be tested.
    """
    df = cls.import_data(data_path)

    try:
        perform_eda(df)
        assert os.path.exists(eda_image_folder)
        logging.info(f"EDA completed. Figures saved to: {eda_image_folder}")
    except Exception as e:
        logging.error(f"EDA failed: {e}")
        raise e


@pytest.mark.xfail()
def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    # Test with valid input
    df_valid = pd.DataFrame({
        'Attrition_Flag': ['Existing Customer', 'Attrited Customer', 'Existing Customer'],
        'Other_Column': [1, 2, 3]
    })
    try:
        X_train, X_test, y_train, y_test = encoder_helper(df_valid)
        # Add assertions here to check if the output is as expected
    except Exception as e:
        pytest.fail(f"Unexpected error occurred with valid input: {e}")

    # Test with invalid input (e.g., incorrect column name)
    df_invalid = pd.DataFrame({
        'Wrong_Column_Name': ['Existing Customer', 'Attrited Customer', 'Existing Customer'],
        'Other_Column': [1, 2, 3]
    })
    with pytest.raises(NonBinaryTargetException):
        encoder_helper(df_invalid)

    # You can add more tests for different types of invalid inputs

def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering
    """
    df = cls.import_data(data_path)
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Gender')
        # Add assertions here to check if the output is as expected
    except Exception as e:
        pytest.fail(f"Unexpected error occurred with valid input: {e}")

    # Test with non-binary target column
    with pytest.raises(NonBinaryTargetException):
        perform_feature_engineering(df, 'Income_Category')


@pytest.mark.xfail()
def test_train_models(train_models):
    """
    test train_models
    """


if __name__ == "__main__":
    run_tests()
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
