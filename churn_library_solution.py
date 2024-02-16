# library doc string


# import libraries
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from constants import (eda_image_folder, model_results_folder,
                       target, cat_columns, quant_columns)
from exceptions import NonBinaryTargetException
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: Path) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv file
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


def encoder_helper(df: pd.DataFrame, category_lst: List,
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
        category_grouped = df.groupby(category)[binary_target].mean()

        # Create a new column in df for enocoded category
        new_column_name = f"{category}_encoded"
        df[new_column_name] = df[category].map(category_grouped)

    return df


def perform_feature_engineering(df: pd.DataFrame, response: str = target) -> (
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]):
    """
    input:
              df: pandas dataframe
              response: target variable

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    # check if response is binary
    unique_vals = df[response].unique()
    if len(unique_vals) == 2:
        # create binary target
        df['binary_target'] = df[response].apply(lambda val: 0 if val == unique_vals[0] else 1)
    else:
        raise NonBinaryTargetException(f"The target column '{response}' is not binary.")
    # check if response is in cat_columns
    if response in cat_columns:
        cat_columns.remove(response)
    df_encoded = encoder_helper(df, cat_columns, binary_target='binary_target')
    cat_columns_encoded = [col + '_encoded' for col in cat_columns if col]
    X = df_encoded[quant_columns + cat_columns_encoded]
    y = df_encoded['binary_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf, model_results_folder):
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
            model_results_folder: folder to store the classification report image

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
    os.makedirs(model_results_folder, exist_ok=True)

    # Save the figure to the specified directory
    try:
        file_path = os.path.join(model_results_folder,
                                 "classification_report.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        print(f"Classification report image saved successfully at {file_path}")
    except Exception as e:
        print(
            f"An error occurred while saving the classification report image: {e}")
        plt.close()


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
    # Todo: test function and add to train
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);


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
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    joblib.dump(lrc, './models/logistic_model.pkl')

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    joblib.dump(lrc, './models/random_forest.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                model_results_folder)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax,
                              alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show()   # Todo: save this file


if __name__ == "__main__":
    df = import_data(Path('data/bank_data.csv'))
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    print("training_models")
    train_models(X_train, X_test, y_train, y_test)
    print('done')
