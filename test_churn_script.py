import os
import logging
import pytest
import churn_library_solution as cls
from datetime import datetime
from constants import data_path, eda_image_path


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


def run_tests():
    logging.info("Starting tests at %s", datetime.now())


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    try:
        df = import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
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

    #import_data.config.cache.set('cache_df', df)
    return df


@pytest.mark.xfail()
def test_eda(perform_eda):
    """
    test perform eda function
    """
    #df = import_data.config.cache.get('cache_df', None)
    try:
        perform_eda(data)
        assert os.path.exists(eda_image_path)
        logging.info(f"EDA completed. Figure saved to: {eda_image_path}")
    except:
        logging.error("EDA failed")


@pytest.mark.xfail()
def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """


@pytest.mark.xfail()
def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """


@pytest.mark.xfail()
def test_train_models(train_models):
    """
    test train_models
    """


if __name__ == "__main__":
    run_tests()
    data = test_import(cls.import_data)
    print(data.head())
    test_eda(cls.perform_eda)
