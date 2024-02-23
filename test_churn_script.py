import os
import logging
import unittest
import joblib
import pytest
import churn_library_solution as cls
from datetime import datetime
from constants import data_path, eda_image_folder, cat_columns, quant_columns, model_path
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
def encoder_helper():
    return cls.encoder_helper


@pytest.fixture(scope="module")
def perform_feature_engineering():
    return cls.perform_feature_engineering


@pytest.fixture(scope="module")
def train_models():
    return cls.train_models


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
        logging.info("SUCCESS: EDA completed. "
                     f"Figures saved to {eda_image_folder}")
    except Exception as e:
        logging.error(f"EDA failed: {e}")
        raise e


def test_encoder_helper(encoder_helper):
    """
    Test the encoder_helper function.
    """

    # Creating data for testing
    df = cls.import_data(data_path)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    test_value = df.groupby('Education_Level')['Churn'].mean()['College']

    # Define the category list
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    category_list = list(set(cols) - set(num_cols))

    # Apply the encoder_helper function
    try:
        df_encoded = encoder_helper(df, category_list, binary_target='Churn')

        # Check if new columns are added correctly
        for category in category_list:
            expected_column = f"{category}_encoded"
            assert expected_column in df_encoded.columns, \
                f"Column {expected_column} not found in DataFrame"

        # Check if the values are calculated correctly
        assert df_encoded['Education_Level_encoded'].dtype == 'float64', \
            "Values are not calculated correctly"
        assert set(df_encoded[df_encoded['Education_Level'] == 'College']
                   ['Education_Level_encoded']) == set([test_value]), \
            "Values are not calculated correctly"

        logging.info("SUCCESS: target encoding")

    except Exception as e:
        logging.error(f"Target encoding failed: {e}")


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering
    """
    df = cls.import_data(data_path)
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Gender')
        # Check if the shape of the data is correct
        assert X_train.shape[1] == len(cat_columns+quant_columns)
        logging.info("SUCCESS: feature engineering completed successfully.")
    except Exception as e:
        pytest.fail(f"Unexpected error occurred with valid input: {e}")

    # Test with non-binary target column
    with pytest.raises(NonBinaryTargetException):
        perform_feature_engineering(df, 'Income_Category')


# def test_train_models(train_models):
#     """
#     Test models are trained and can make predictions.
#     """
#     full_data = cls.import_data(data_path)
#     test_data = full_data.sample(frac=0.1, random_state=42)
#     X_train, X_test, y_train, y_test = cls.perform_feature_engineering(test_data)
#
#     # test train_models
#     train_models(X_train, X_test, y_train, y_test)
#
#     # Assert models were trained and saved correctly
#     logistic_regression_model_path = os.path.join(model_path, "logistic_regression_classifier.pkl")
#     random_forest_model_path = os.path.join(model_path, "random_forest_classifier.pkl")
#
#     assert os.path.exists(logistic_regression_model_path), "Logistic regression model file not found."
#     assert os.path.exists(random_forest_model_path), "Random forest model file not found."
#
#     # Load models and assert they can make predictions
#     lrc = joblib.load(logistic_regression_model_path)
#     cv_rfc = joblib.load(random_forest_model_path)
#
#     assert lrc.predict(X_test) is not None, "Logistic regression model failed to make predictions."
#     assert cv_rfc.predict(X_test) is not None, "Random forest model failed to make predictions."
#
#     logging.info("SUCCESS: Models trained and predictions made successfully.")


def test_train_models(train_models):
    """
    Test models are trained and can make predictions.
    """
    try:
        full_data = cls.import_data(data_path)
        test_data = full_data.sample(frac=0.1, random_state=42)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(test_data)

        # test train_models
        train_models(X_train, X_test, y_train, y_test)

        # Assert models were trained and saved correctly
        logistic_regression_model_path = os.path.join(model_path, "logistic_regression_classifier.pkl")
        random_forest_model_path = os.path.join(model_path, "random_forest_classifier.pkl")

        assert os.path.exists(logistic_regression_model_path), "Logistic regression model file not found."
        assert os.path.exists(random_forest_model_path), "Random forest model file not found."

        # Load models and assert they can make predictions
        lrc = joblib.load(logistic_regression_model_path)
        cv_rfc = joblib.load(random_forest_model_path)

        assert lrc.predict(X_test) is not None, "Logistic regression model failed to make predictions."
        assert cv_rfc.predict(X_test) is not None, "Random forest model failed to make predictions."

        logging.info("SUCCESS: Models trained and predictions made successfully.")
    except AssertionError as e:
        logging.error(f"Assertion error: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    run_tests()
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
