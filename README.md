# Customer Churn Prediction Project

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict customer churn using a dataset from a bank. It involves exploratory data analysis (EDA), feature engineering, and training machine learning models to classify customers as churned or not churned. The project uses Python and several libraries, including pandas, numpy, matplotlib, seaborn, and scikit-learn.

## Project Structure
- `churn_library_solution.py`: contains the functions used in the churn script  
- `test_churn_script.py`: contains the unit tests for the functions in churn_library_solution.py
- `constants.py`: Contains constants used across the project, such as file paths and column names.
- `data/`: Directory containing the dataset used for the project.
- `models/`: Directory where trained models are saved.
- `images/eda/`: Directory where EDA images are saved.
- `images/results/`: Directory where model results and feature importance images are saved.

## Running Files
In terminal, run the following command: 
```bash
python churn_library_solution.py
```
This will perform the following steps:
- Import the dataset.
- Perform exploratory data analysis and save the figures.
- Encode categorical features and split the data into training and testing sets.
- Train a Logistic Regression model and a Random Forest Classifier.
- Evaluate the models and save classification reports, ROC curves, and feature importance plots.
 
## For testing:
```bash
python test_churn_script.py 
``` 
- `logs/`: Running the test script creates a log file in

## Testing with Pytest:
```bash
pytest test_churn_script.py
```


## Logging

### Log File Overview

The library uses Python's built-in `logging` module to log its operations. Logs are saved to `./logs/churn_library.log`, providing a detailed record of the library's execution, including successful operations and any errors or warnings encountered.

#### Examples of Log Messages

- **INFO**: Indicates successful execution of a function.
  ```
  root - INFO - SUCCESS: data imported successfully
  ```
- **ERROR**: Indicates an error occurred during execution.
  ```
  root - ERROR - Target encoding failed: 'Churn' column not found
  ```

Reviewing the log file can help diagnose issues, verify successful operations, and understand the library's workflow.
