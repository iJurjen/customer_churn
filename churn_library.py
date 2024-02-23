""" This module contains functions for
EDA
Feature Engineering (including encoding of categorical variables)
Model Training
Prediction
Model Evaluation
"""


# import libraries
import os
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from constants import (eda_image_folder, model_path, model_results_folder,
                       target, cat_columns, quant_columns)
from exceptions import NonBinaryTargetException


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: Path) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv file
    output:
            data: pandas dataframe
    """
    data = pd.read_csv(pth)
    return data


def save_plot(
        data,
        column,
        plot_type,
        file_name,
        folder=eda_image_folder,
        **kwargs):
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
            sns.histplot(data[column], **kwargs)
        else:
            data[column].hist(**kwargs)
    elif plot_type == 'bar':
        data[column].value_counts('normalize').plot(kind='bar', **kwargs)
    elif plot_type == 'heatmap':
        sns.heatmap(data.corr(), **kwargs)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, file_name))
    plt.close()


def perform_eda(data: pd.DataFrame) -> None:
    """
    Perform EDA on df and save figures to images folder.

    Parameters:
    df (pd.DataFrame): pandas dataframe.
    """
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        save_plot(data, 'Churn', 'hist', 'churn_distribution.png')
        save_plot(
            data,
            'Customer_Age',
            'hist',
            'customer_age_distribution.png')
        save_plot(data, 'Marital_Status', 'bar',
                  'marital_status_distribution.png')
        save_plot(data, 'Total_Trans_Ct', 'hist',
                  'total_transaction_distribution.png', stat='density',
                  kde=True)
        save_plot(data, None, 'heatmap', 'heatmap.png', annot=False,
                  cmap='Dark2_r', linewidths=2)
    except Exception as error:
        raise error


def encoder_helper(data: pd.DataFrame, category_lst: List,
                   binary_target: str) -> pd.DataFrame:
    """
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    category_lst (list): List of columns that contain categorical features.
    binary_target (str): Name of the target column

    Returns:
    pandas.DataFrame: DataFrame with new columns for each categorical feature,
                      representing the proportion of the response.
    """
    # Iterate over each categorical column
    for category in category_lst:
        # Group by the category and calculate the mean of the response
        category_grouped = data.groupby(category)[binary_target].mean()

        # Create a new column in df for enocoded category
        new_column_name = f"{category}_encoded"
        data[new_column_name] = data[category].map(category_grouped)

    return data


def perform_feature_engineering(data: pd.DataFrame,
                                response: str = target) -> (Tuple[pd.DataFrame,
                                                                  pd.DataFrame,
                                                                  pd.Series,
                                                                  pd.Series]):
    """
    input:
              df: pandas dataframe
              response: target variable

    output:
              train_data: encoded_data training data
              test_data: encoded_data testing data
              y_train: binary_target training data
              y_test: binary_target testing data
    """
    # check if response is binary
    unique_vals = data[response].unique()
    if len(unique_vals) == 2:
        # create binary target
        data['binary_target'] = data[response].apply(
            lambda val: 0 if val == unique_vals[0] else 1)
    else:
        raise NonBinaryTargetException(
            f"The target column '{response}' is not binary.")
    # check if response is in cat_columns
    if response in cat_columns:
        cat_columns.remove(response)
    df_encoded = encoder_helper(
        data, cat_columns, binary_target='binary_target')
    cat_columns_encoded = [col + '_encoded' for col in cat_columns if col]
    encoded_data = df_encoded[quant_columns + cat_columns_encoded]
    binary_target = df_encoded['binary_target']
    train_data, test_data, y_train, y_test = train_test_split(
        encoded_data, binary_target, test_size=0.3, random_state=42)
    return train_data, test_data, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf, results_folder):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            results_folder: folder to store the classification report image

    output:
             None
    """
    # Create a figure for the classification reports
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.1, right=0.8, top=0.75,
                        bottom=0.25)  # Adjust the top and bottom

    # Add Logistic Regression Training classification report text to subplot
    lr_train_report = classification_report(y_train, y_train_preds_lr)
    fig.text(0.01, 0.3, 'Logistic Regression Train\n' + lr_train_report,
             fontfamily='monospace')  # Adjust the vertical position

    # Add Logistic Regression Testing classification report text to subplot
    lr_test_report = classification_report(y_test, y_test_preds_lr)
    fig.text(0.01, 0.1, 'Logistic Regression Test\n' + lr_test_report,
             fontfamily='monospace')  # Adjust the vertical position

    # Add Random Forest Training classification report text to subplot
    rf_train_report = classification_report(y_train, y_train_preds_rf)
    fig.text(0.01, 0.9, 'Random Forest Train\n' + rf_train_report,
             fontfamily='monospace')  # Adjust the vertical position

    # Add Random Forest Testing classification report text to subplot
    rf_test_report = classification_report(y_test, y_test_preds_rf)
    fig.text(0.01, 0.7, 'Random forest Test\n' + rf_test_report,
             fontfamily='monospace')  # Adjust the vertical position

    # Remove axes for a clean look
    plt.axis('off')

    # Ensure the directory for saving the image exists
    os.makedirs(results_folder, exist_ok=True)

    # Save the figure to the specified directory
    try:
        file_path = os.path.join(results_folder,
                                 "classification_report.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        print(f"Classification report image saved successfully at {file_path}")
    except Exception as error:
        print(
            f"An error occurred while saving the classification report image: {error}")
        plt.close()


def feature_importance_plot(model, train_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [train_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(train_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(train_data.shape[1]), names, rotation=90)

    # Save the figure to the specified directory
    try:
        plt.savefig(output_pth, bbox_inches='tight')
        plt.close()
        print(
            f"Classification report image saved successfully at {output_pth}")
    except Exception as error:
        print(
            f"An error occurred while saving the classification report image: {error}")
        plt.close()


def train_models(train_data, test_data, y_train, y_test):
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

    # train and store Logistic Regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(train_data, y_train)
    joblib.dump(lrc, model_path + 'logistic_regression_classifier.pkl')

    # train and store Random Forest model
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(train_data, y_train)
    joblib.dump(cv_rfc, model_path + 'random_forest_classifier.pkl')

    # predictions logistic regression model
    y_train_preds_lr = lrc.predict(train_data)
    y_test_preds_lr = lrc.predict(test_data)

    # predictions random forest model
    y_train_preds_rf = cv_rfc.predict(train_data)
    y_test_preds_rf = cv_rfc.predict(test_data)

    # create classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                model_results_folder)

    # create ROC curves
    lrc_plot = plot_roc_curve(lrc, test_data, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc, test_data, y_test, ax=ax,
                              alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    try:
        file_path = os.path.join(model_results_folder,
                                 "ROC_curves.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved successfully at {file_path}")
    except Exception as error:
        print(
            f"An error occurred while saving the classification report image: {error}")
        plt.close()

    # create feature importance plot
    output_pth = os.path.join(model_results_folder, "Feature_importance.png")
    feature_importance_plot(cv_rfc, train_data, output_pth)


if __name__ == "__main__":
    bank_data = import_data(Path('data/bank_data.csv'))
    perform_eda(bank_data)
    bank_data_train, bank_data_test, train_labels, test_labels = perform_feature_engineering(
        bank_data)
    print("training_models")
    train_models(bank_data_train, bank_data_test, train_labels, test_labels)
    print('done')
