"""This file contains constants for the churn library and test script"""

data_path = "./data/bank_data.csv"
eda_image_folder = "./images/eda/"
model_results_folder = "./images/results"
# model_path = "./models/churn_model.pkl"

# Target variable
target = 'Attrition_Flag'
# List of categorical variables
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category',
    'Attrition_Flag'
]
# List of numerical variables
quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]
