# library doc string


# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import eda_image_folder
from pathlib import Path


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: Path) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def save_plot(df, column, plot_type, file_name, folder=eda_image_folder, **kwargs):
    """
    Helper function to create and save a plot.

    Parameters:
    df (pd.DataFrame): The dataframe to plot.
    column (str): The column to plot.
    plot_type (str): Type of plot (e.g., 'hist', 'bar', 'heatmap').
    file_name (str): Name of the file to save the plot.
    folder (str): Folder path to save the plot.
    **kwargs: Additional keyword arguments for the plotting function.
    """
    plt.figure(figsize=(20, 10))

    if plot_type == 'hist':
        # Check if 'stat' is in kwargs for seaborn histplot
        if 'stat' in kwargs:
            sns.histplot(df[column], **kwargs)
        else:
            df[column].hist(**kwargs)
    elif plot_type == 'bar':
        df[column].value_counts('normalize').plot(kind='bar', **kwargs)
    elif plot_type == 'heatmap':
        sns.heatmap(df.corr(), **kwargs)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, file_name))
    plt.close()


def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform EDA on df and save figures to images folder.

    Parameters:
    df (pd.DataFrame): pandas dataframe.
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        save_plot(df, 'Churn', 'hist', 'churn_distribution.png')
        save_plot(df, 'Customer_Age', 'hist', 'customer_age_distribution.png')
        save_plot(df, 'Marital_Status', 'bar',
                  'marital_status_distribution.png')
        save_plot(df, 'Total_Trans_Ct', 'hist',
                  'total_transaction_distribution.png', stat='density',
                  kde=True)
        save_plot(df, None, 'heatmap', 'heatmap.png', annot=False,
                  cmap='Dark2_r', linewidths=2)
    except Exception as e:
        raise e


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    pass


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass
