from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# import pandas as pd
# from imblearn.over_sampling import SMOTE
def frequency_encoding(train_data, test_data):
    """
    Apply SMOTE to the DataFrame to handle class imbalance,
    while performing frequency encoding for categorical columns.

    Parameters:
    train_data (DataFrame): The input training DataFrame containing features and labels.
    test_data (DataFrame): The input testing DataFrame containing features and labels.

    Returns:
    DataFrame: The training DataFrame with resampled (balanced) data and frequency encoding applied.
    DataFrame: The testing DataFrame with frequency encoding applied.
    """

    # Create copies of the training and testing DataFrames to avoid modifying the original data
    train_encoded = train_data.copy()
    test_encoded = test_data.copy()
    
    # Initialize a dictionary to store encodings for each column
    encoding_dicts = {}
    
    # Iterate over each column in training data
    for column in train_data.columns:
        if train_data[column].dtype == 'object':
            # Replace categorical values with their corresponding frequencies
            encoding = train_data.groupby(column).size() / len(train_data)
            # Store the encoding dictionary for this column
            encoding_dicts[column] = encoding
            train_data[column] = train_data[column].map(encoding)
    
    # Apply frequency encoding to categorical columns in test data using encoding values from training data
    for column in test_data.columns:
        if test_data[column].dtype == 'object':
            test_data[column] = test_data[column].map(encoding_dicts[column])


    return 

#%%
result = frequency_encoding(df_random_prep_logically, df_testing_random_prep_logically)
result = frequency_encoding(df_percentage_prep_logically,df_testing_percentage_prep_logically)
result = frequency_encoding(df_random_prep_correlation,df_testing_random_prep_correlation)
result = frequency_encoding(df_percentage_prep_correlation, df_testing_percentage_prep_correlation)

#%%


