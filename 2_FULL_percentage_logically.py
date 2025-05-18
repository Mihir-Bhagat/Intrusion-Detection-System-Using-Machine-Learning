# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:48:02 2024

@author: arsla
"""


'''
issues- the order of us applying the preprocessing to proto values and state values would make a difference in the answer, 
howerver it was done with proto as 1st since logically it is more related as a transactrion protocol 
being executed is more likely to determine the presence of an attack

'''

#%%
#Method 1: Establishing pre-processing with Most Logically Related Columns

#%%
#Type 2: percentage based Assigning of values
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_training-set.csv",index_col=0)

df2 = df.copy()

# # Filter DataFrame to include only rows where 'service' is '-'
# combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_proto_state_with_state_hyphen)


def filter_and_group_by_service(df, group_by_columns):
    """
    Filter DataFrame to include only rows where a specified column is '-' and group by specified columns.

    Parameters:
    - df: DataFrame to filter and group.
    - group_by_columns: List of column names to group by.

    Returns:
    - DataFrame containing filtered and grouped data.
    """
    filtered_data = df[df['service'] == '-']
    grouped_data = filtered_data.groupby(group_by_columns + ['service']).size().reset_index().rename(columns={0: 'count'})
    grouped_data = grouped_data.sort_values(by='count', ascending=False)
    return grouped_data

combinations_proto_state_with_state_hyphen = filter_and_group_by_service(df2, ['proto', 'state'])
print(combinations_proto_state_with_state_hyphen)


# # an empty dictionary to store the count of occurrences of each service value
# service_count_dict = {}

# # Iterate over each unique service value
# for service_value in df2['service'].unique():
#     # Skip the '-' value
#     if service_value == '-':
#         continue

#     # Count the occurrences of the current service value in the DataFrame
#     count = df2[df2['service'] == service_value].shape[0]

#     # Add the service value and its count to the dictionary
#     service_count_dict[service_value] = count

# '''
# making a dictionary with combinations of proto and state with their service values other than '-'
# '''

# # Initialize an empty dictionary to store unique service values for each combination of proto and state
# unique_service_values_dict = {}

# # Iterate over each unique combination of 'proto' and 'state'
# for index, row in combinations_proto_state_with_state_hyphen.iterrows():
#     # Get the values of 'proto' and 'state' for the current row
#     current_proto = row['proto']
#     current_state = row['state']

#     # Filter df2 for rows matching the current combination of 'proto' and 'state', excluding rows where 'service' is '-'
#     filtered_rows = df2[(df2['proto'] == current_proto) & (df2['state'] == current_state) & (df2['service'] != '-')]

#     # Extract unique values in the 'service' column for the filtered rows
#     unique_service_values = filtered_rows['service'].unique()

#     # Store the unique service values in the dictionary
#     unique_service_values_dict[(current_proto, current_state)] = set(unique_service_values)

# # Print the dictionary'
# print(unique_service_values_dict)

# # Create a new dictionary to store non-empty values
# non_empty_service_values_dict = {}

# # Iterate over the items in the original dictionary
# for key, value in unique_service_values_dict.items():
#     # Check if the set of unique service values is not empty
#     if value:
#         # Add the key-value pair to the new dictionary
#         non_empty_service_values_dict[key] = value

# # Print the filtered dictionary
# print(non_empty_service_values_dict)

len(df.columns)
def create_unique_service_values_dict(df, combinations_df):
    """
    Create a dictionary with combinations of proto and state with their non-empty service values.

    Parameters:
    - df: DataFrame to extract unique service values from.
    - combinations_df: DataFrame containing combinations of 'proto' and 'state' with hyphen in the 'service' column.

    Returns:
    - Dictionary containing combinations of proto and state as keys and non-empty service values as values.
    """
    # Count occurrences of each service value
    service_count_dict = {}
    for service_value in df['service'].unique():
        if service_value != '-':
            count = df[df['service'] == service_value].shape[0]
            service_count_dict[service_value] = count

    # Create dictionary with combinations of proto and state with their service values other than '-'
    unique_service_values_dict = {}
    for index, row in combinations_df.iterrows():
        current_proto = row['proto']
        current_state = row['state']
        filtered_rows = df[(df['proto'] == current_proto) & (df['state'] == current_state) & (df['service'] != '-')]
        unique_service_values = set(filtered_rows['service'].unique())
        unique_service_values_dict[(current_proto, current_state)] = unique_service_values

    # Filter out empty service value sets
    non_empty_service_values_dict = {key: value for key, value in unique_service_values_dict.items() if value}
    
    return non_empty_service_values_dict

# Create dictionary with combinations of proto and state with their non-empty service values
non_empty_service_values_dict = create_unique_service_values_dict(df2, combinations_proto_state_with_state_hyphen)
print(non_empty_service_values_dict)


#here we measure their relative appearance in the combination and assign them respectively

# Initialize an empty dictionary to store relative occurrence frequencies of unique service values
relative_occurrence_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)]

    # Calculate the relative occurrence frequency of each unique service value
    total_count = len(specific_rows)
    service_counts = specific_rows['service'].value_counts(normalize=True)

    # Store the relative occurrence frequencies in the dictionary
    relative_occurrence_dict[key] = service_counts.to_dict()

# Print the dictionary containing relative occurrence frequencies
print(relative_occurrence_dict)

# Initialize an empty dictionary to store the counts of unique service values
service_counts_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)]

    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()

    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()

# Print the dictionary containing service counts
print(service_counts_dict)

# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)]

    # Get the count of '-' if it exists
    count_dash = value.get('-', 0)

    # Remove '-' from the counts dictionary
    if '-' in value:
        del value['-']

    # Calculate the total count of service values excluding '-'
    total_count = sum(value.values())

    # Calculate the weights and corresponding service values
    weights = [count / total_count for count in value.values()]
    service_values = list(value.keys())

    # Randomly assign the service values based on the calculated weights
    assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)

    # Assign the selected values to the DataFrame
    specific_rows['service'] = assigned_values

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows


#%%
'''
Checking for what valiues with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)

df['service'].value_counts()
#%%
# #iteratinng for values that have no other previous history- based on the frequency of occurrennce in the whole data

# # Print the dictionary containing the count of occurrences of each service value
# print(service_count_dict)

# # Replace '-' in the 'service' column with values based on their frequencies
# for index, row in df2.iterrows():
#     if row['service'] == '-':
#         # Generate a weighted list of service values based on frequencies
#         service_values, frequencies = zip(*service_count_dict.items())
#         service = np.random.choice(service_values, p=np.array(frequencies) / sum(frequencies))
#         df2.at[index, 'service'] = service

# # Print the updated DataFrame
# print(df2)

# #assigning the updated dataframe
# df_weighted_preprocessing_using_proto_state=df2.copy()

#%%
combinations_proto_with_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_with_hyphen = combinations_proto_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_proto_with_hyphen)
print('We see that there are about',combinations_proto_with_hyphen[combinations_proto_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")


unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_proto_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_proto = row['proto']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['proto'] == current_proto) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_proto)] = set(unique_service_values)

# Print the dictionary
print(unique_service_values_dict)

# Create a new dictionary to store non-empty values
non_empty_service_values_dict = {}

# Iterate over the items in the original dictionary
for key, value in unique_service_values_dict.items():
    # Check if the set of unique service values is not empty
    if value:
        # Add the key-value pair to the new dictionary
        non_empty_service_values_dict[key] = value


service_counts_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    proto_value= key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['proto'] == proto_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()
# Print the dictionary containing service counts
print(service_counts_dict)


# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    proto_value= key
    specific_rows = df2[(df2['proto'] == proto_value)]
    
    # Get the count of '-' if it exists
    count_dash = value.get('-', 0)
    
    # Remove '-' from the counts dictionary
    if '-' in value:
        del value['-']
    
    # Calculate the total count of service values excluding '-'
    total_count = sum(value.values())
    
    # Calculate the weights and corresponding service values
    weights = [count / total_count for count in value.values()]
    service_values = list(value.keys())
    
    # Randomly assign the service values based on the calculated weights
    assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
    
    # Assign the selected values to the DataFrame
    specific_rows['service'] = assigned_values
    
    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    
combinations_proto_with_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_with_hyphen = combinations_proto_with_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_with_hyphen)
print('We see that there are about',combinations_proto_with_hyphen[combinations_proto_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

#%%
'''
Checking for what valiues with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)
#%%
combinations_state_with_hyphen = df2[df2['service'] == '-'].groupby(['state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_state_with_hyphen = combinations_state_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_state_with_hyphen)
print('We see that there are about',combinations_state_with_hyphen[combinations_state_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")


unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_state_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_state = row['state']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['state'] == current_state) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_state)] = set(unique_service_values)

# Print the dictionary
print(unique_service_values_dict)

# Create a new dictionary to store non-empty values
non_empty_service_values_dict = {}

# Iterate over the items in the original dictionary
for key, value in unique_service_values_dict.items():
    # Check if the set of unique service values is not empty
    if value:
        # Add the key-value pair to the new dictionary
        non_empty_service_values_dict[key] = value


service_counts_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    state_value= key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['state'] == state_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()
# Print the dictionary containing service counts
print(service_counts_dict)


# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    state_value= key
    specific_rows = df2[(df2['state'] == state_value)]
    
    # Get the count of '-' if it exists
    count_dash = value.get('-', 0)
    
    # Remove '-' from the counts dictionary
    if '-' in value:
        del value['-']
    
    # Calculate the total count of service values excluding '-'
    total_count = sum(value.values())
    
    # Calculate the weights and corresponding service values
    weights = [count / total_count for count in value.values()]
    service_values = list(value.keys())
    
    # Randomly assign the service values based on the calculated weights
    assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
    
    # Assign the selected values to the DataFrame
    specific_rows['service'] = assigned_values
    
    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    
combinations_state_with_hyphen = df2[df2['service'] == '-'].groupby(['state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_state_with_hyphen = combinations_state_with_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_state_with_hyphen)
print('We see that there are about',combinations_state_with_hyphen[combinations_state_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")
#%%
'''
Checking for what valiues with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)
#%%
print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values\nwhich makes no sense as neither their proto value (i.e. icmp) nor their individual state values(i.e ECO,PAR,URN,no) has any past history of service value other than '-'\nthus we would drop them as the total is very less compared to the whole dataset")
# Drop rows with '-' in the 'service' column
df2 = df2[df2['service'] != '-']

#%%
#storing in a variable
df_percentage_prep_logically=df2.copy()
# df_percentage_prep_logically.columns
#%%
#checking for distribution of service values in each type 1 and type 2
df['service'].value_counts()
print(df_random_prep_logically['service'].value_counts())
print(df_percentage_prep_logically['service'].value_counts())



#%%%


    

#For Test Data


#%%


'''
issues- the order of us applying the preprocessing to proto values and state values would make a difference in the answer, 
howerver it was done with proto as 1st since logically it is more related as a transactrion protocol 
being executed is more likely to determine the presence of an attack

'''

#%%
#Method 1: Establishing pre-processing with Most Logically Related Columns

#%%
#Type 2: percentage based Assigning of values
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_testing-set.csv",index_col=0)
df2 = df.copy()

# Filter DataFrame to include only rows where 'service' is '-'
combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_proto_state_with_state_hyphen)

# an empty dictionary to store the count of occurrences of each service value
service_count_dict = {}

# Iterate over each unique service value
for service_value in df2['service'].unique():
    # Skip the '-' value
    if service_value == '-':
        continue

    # Count the occurrences of the current service value in the DataFrame
    count = df2[df2['service'] == service_value].shape[0]

    # Add the service value and its count to the dictionary
    service_count_dict[service_value] = count

'''
making a dictionary with combinations of proto and state with their service values other than '-'
'''

# Initialize an empty dictionary to store unique service values for each combination of proto and state
unique_service_values_dict = {}

# Iterate over each unique combination of 'proto' and 'state'
for index, row in combinations_proto_state_with_state_hyphen.iterrows():
    # Get the values of 'proto' and 'state' for the current row
    current_proto = row['proto']
    current_state = row['state']

    # Filter df2 for rows matching the current combination of 'proto' and 'state', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['proto'] == current_proto) & (df2['state'] == current_state) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_proto, current_state)] = set(unique_service_values)

# Print the dictionary'
print(unique_service_values_dict)

# Create a new dictionary to store non-empty values
non_empty_service_values_dict = {}

# Iterate over the items in the original dictionary
for key, value in unique_service_values_dict.items():
    # Check if the set of unique service values is not empty
    if value:
        # Add the key-value pair to the new dictionary
        non_empty_service_values_dict[key] = value

# Print the filtered dictionary
print(non_empty_service_values_dict)

#here we measure their relative appearance in the combination and assign them respectively

# Initialize an empty dictionary to store relative occurrence frequencies of unique service values
relative_occurrence_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)]

    # Calculate the relative occurrence frequency of each unique service value
    total_count = len(specific_rows)
    service_counts = specific_rows['service'].value_counts(normalize=True)

    # Store the relative occurrence frequencies in the dictionary
    relative_occurrence_dict[key] = service_counts.to_dict()

# Print the dictionary containing relative occurrence frequencies
print(relative_occurrence_dict)

# Initialize an empty dictionary to store the counts of unique service values
service_counts_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)]

    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()

    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()

# Print the dictionary containing service counts
print(service_counts_dict)

# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)]

    # Get the count of '-' if it exists
    count_dash = value.get('-', 0)

    # Remove '-' from the counts dictionary
    if '-' in value:
        del value['-']

    # Calculate the total count of service values excluding '-'
    total_count = sum(value.values())

    # Calculate the weights and corresponding service values
    weights = [count / total_count for count in value.values()]
    service_values = list(value.keys())

    # Randomly assign the service values based on the calculated weights
    assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)

    # Assign the selected values to the DataFrame
    specific_rows['service'] = assigned_values

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows


#%%
'''
Checking for what valiues with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)
#%%
# #iteratinng for values that have no other previous history- based on the frequency of occurrennce in the whole data

# # Print the dictionary containing the count of occurrences of each service value
# print(service_count_dict)

# # Replace '-' in the 'service' column with values based on their frequencies
# for index, row in df2.iterrows():
#     if row['service'] == '-':
#         # Generate a weighted list of service values based on frequencies
#         service_values, frequencies = zip(*service_count_dict.items())
#         service = np.random.choice(service_values, p=np.array(frequencies) / sum(frequencies))
#         df2.at[index, 'service'] = service

# # Print the updated DataFrame
# print(df2)

# #assigning the updated dataframe
# df_weighted_preprocessing_using_proto_state=df2.copy()

#%%
combinations_proto_with_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_with_hyphen = combinations_proto_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_proto_with_hyphen)
print('We see that there are about',combinations_proto_with_hyphen[combinations_proto_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")


unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_proto_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_proto = row['proto']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['proto'] == current_proto) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_proto)] = set(unique_service_values)

# Print the dictionary
print(unique_service_values_dict)

# Create a new dictionary to store non-empty values
non_empty_service_values_dict = {}

# Iterate over the items in the original dictionary
for key, value in unique_service_values_dict.items():
    # Check if the set of unique service values is not empty
    if value:
        # Add the key-value pair to the new dictionary
        non_empty_service_values_dict[key] = value


service_counts_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    proto_value= key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['proto'] == proto_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()
# Print the dictionary containing service counts
print(service_counts_dict)


# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    proto_value= key
    specific_rows = df2[(df2['proto'] == proto_value)]
    
    # Get the count of '-' if it exists
    count_dash = value.get('-', 0)
    
    # Remove '-' from the counts dictionary
    if '-' in value:
        del value['-']
    
    # Calculate the total count of service values excluding '-'
    total_count = sum(value.values())
    
    # Calculate the weights and corresponding service values
    weights = [count / total_count for count in value.values()]
    service_values = list(value.keys())
    
    # Randomly assign the service values based on the calculated weights
    assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
    
    # Assign the selected values to the DataFrame
    specific_rows['service'] = assigned_values
    
    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    
combinations_proto_with_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_with_hyphen = combinations_proto_with_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_with_hyphen)
print('We see that there are about',combinations_proto_with_hyphen[combinations_proto_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

#%%
'''
Checking for what valiues with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)
#%%
combinations_state_with_hyphen = df2[df2['service'] == '-'].groupby(['state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_state_with_hyphen = combinations_state_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_state_with_hyphen)
print('We see that there are about',combinations_state_with_hyphen[combinations_state_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")


unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_state_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_state = row['state']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['state'] == current_state) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_state)] = set(unique_service_values)

# Print the dictionary
print(unique_service_values_dict)

# Create a new dictionary to store non-empty values
non_empty_service_values_dict = {}

# Iterate over the items in the original dictionary
for key, value in unique_service_values_dict.items():
    # Check if the set of unique service values is not empty
    if value:
        # Add the key-value pair to the new dictionary
        non_empty_service_values_dict[key] = value


service_counts_dict = {}

# Iterate over each key-value pair in the non-empty service values dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    state_value= key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['state'] == state_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()
# Print the dictionary containing service counts
print(service_counts_dict)


# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    state_value= key
    specific_rows = df2[(df2['state'] == state_value)]
    
    # Get the count of '-' if it exists
    count_dash = value.get('-', 0)
    
    # Remove '-' from the counts dictionary
    if '-' in value:
        del value['-']
    
    # Calculate the total count of service values excluding '-'
    total_count = sum(value.values())
    
    # Calculate the weights and corresponding service values
    weights = [count / total_count for count in value.values()]
    service_values = list(value.keys())
    
    # Randomly assign the service values based on the calculated weights
    assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
    
    # Assign the selected values to the DataFrame
    specific_rows['service'] = assigned_values
    
    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    
combinations_state_with_hyphen = df2[df2['service'] == '-'].groupby(['state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_state_with_hyphen = combinations_state_with_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_state_with_hyphen)
print('We see that there are about',combinations_state_with_hyphen[combinations_state_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")
#%%
'''
Checking for what valiues with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)
#%%
print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values\nwhich makes no sense as neither their proto value (i.e. icmp) nor their individual state values(i.e ECO,PAR,URN,no) has any past history of service value other than '-'\nthus we would drop them as the total is very less compared to the whole dataset")
# Drop rows with '-' in the 'service' column
df2 = df2[df2['service'] != '-']


#%%
#storing in a variable
df_testing_percentage_prep_logically=df2.copy()
print(df_testing_percentage_prep_logically['is_sm_ips_ports'].value_counts())

df_testing_percentage_prep_logically.to_csv('preprocessed_test_data.csv')


df_testing_percentage_prep_logically.info()
