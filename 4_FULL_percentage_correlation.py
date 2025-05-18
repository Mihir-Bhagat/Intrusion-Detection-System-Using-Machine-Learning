

'''
issues- the order of us applying the preprocessing to dwin values and service values would make a difference in the answer, 
howerver it was done with proto as 1st since its most correlated comparatively
'''

#%%
#Method 2: Establishing pre-processing with Most correlated Columns 

#%%
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_training-set.csv",index_col=0)
df.columns
df2 = df.copy()
# label_encoding all the object columns
from sklearn.preprocessing import LabelEncoder
# Create a dictionary to store mappings
label_mappings = {} 
# Iterate over columns
for column in df.columns:
    # Check if column has object dtype
    if df2[column].dtype == 'object':
        # Apply label encoding
        label_encoder = LabelEncoder()
        df2[column] = label_encoder.fit_transform(df2[column])
        # Store mappings
        label_mappings[column] = {index: label for index, label in enumerate(label_encoder.classes_)}
        
#getting top two highly correlated columns
# Calculate correlation matrixm
correlation_matrix = df2.corr()


'''
checking for correlation between service and all other columns
'''
service_correlation = df2.corr()['service']

print('top 5 correlated columns with service:\n',service_correlation.drop('service').sort_values(ascending=False).head(5))

#theres a considerable difference between the ----1st, 2nd---- and----the rest of the columns
#%%

#Type 2: Weighted Assigning of values 
#%%

df2 = df.copy()
# Filter DataFrame to include only rows where 'service' is '-'

combinations_dwin_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_swin_with_hyphen = combinations_dwin_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_swin_with_hyphen)

unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_dwin_swin_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_dwin = row['dwin']
    current_swin = row['swin']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['dwin'] == current_dwin) & (df2['swin'] == current_swin) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered 
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_dwin, current_swin)] = set(unique_service_values)

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
    dwin_value, swin_value = key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['dwin'] == dwin_value) & (df2['swin'] == swin_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()

# Print the dictionary containing service counts
print(service_counts_dict)
# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    dwin_value, swin_value = key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['dwin'] == dwin_value) & (df2['swin'] == swin_value)]
    
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

    # Assign the selected values to the DataFrame using .loc[]
    df2.loc[specific_rows.index, 'service'] = assigned_values

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows

for key, value in service_counts_dict.items():
    # Get the dwin value from the key
    dwin_value, _ = key
    
    # Filter DataFrame to include only rows where 'dwin' matches the current key
    specific_rows_index = df2[(df2['dwin'] == dwin_value)].index
    
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
    assigned_values = np.random.choice(service_values, size=len(specific_rows_index), p=weights)
    
    # Assign the selected values to the DataFrame using .loc
    df2.loc[specific_rows_index, 'service'] = assigned_values

#checking for values remaining after preprocessing
combinations_dwin_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_swin_with_hyphen = combinations_dwin_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_swin_with_hyphen)

print('We see that there are about',combinations_dwin_swin_with_hyphen[combinations_dwin_swin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")
# # Drop the rows where the 'service' column contains '-'
# df2 = df2[df2['service'] != '-']
#%%
combinations_dwin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_with_hyphen = combinations_dwin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_with_hyphen)
print('We see that there are about',combinations_dwin_with_hyphen[combinations_dwin_swin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")


unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_dwin_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_dwin = row['dwin']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['dwin'] == current_dwin) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_dwin)] = set(unique_service_values)

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
    dwin_value= key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['dwin'] == dwin_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()
# Print the dictionary containing service counts
print(service_counts_dict)


# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    dwin_value= key
    specific_rows = df2[(df2['dwin'] == dwin_value)]
    
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
    
combinations_dwin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_with_hyphen = combinations_dwin_with_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_dwin_with_hyphen)
print('We see that there are about',combinations_dwin_with_hyphen[combinations_dwin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")





#%%
combinations_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_swin_with_hyphen = combinations_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_swin_with_hyphen)

unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_swin_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_swin = row['swin']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['swin'] == current_swin) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered rows
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_swin)] = set(unique_service_values)

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
    swin_value= key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['swin'] == swin_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()
# Print the dictionary containing service counts
print(service_counts_dict)


# # Iterate over each key-value pair in the service counts dictionary
# for key, value in service_counts_dict.items():
#     swin_value= key
#     specific_rows = df2[(df2['swin'] == swin_value)]
    
#     # Get the count of '-' if it exists
#     count_dash = value.get('-', 0)
    
#     # Remove '-' from the counts dictionary
#     if '-' in value:
#         del value['-']
    
#     # Calculate the total count of service values excluding '-'
#     total_count = sum(value.values())
    
#     # Calculate the weights and corresponding service values
#     weights = [count / total_count for count in value.values()]
#     service_values = list(value.keys())
    
#     # Randomly assign the service values based on the calculated weights
#     assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
        
#     # Assign the selected values to the DataFrame using .loc[]
#     df2.loc[specific_rows.index, 'service'] = assigned_values

    
#     # Update the specific rows back into the main DataFrame df2
#     df2.loc[specific_rows.index] = specific_rows

for key, value in service_counts_dict.items():
    # Get the dwin value from the key
    swin_value= key
    
    # Filter DataFrame to include only rows where 'dwin' matches the current key
    specific_rows_index = df2[(df2['swin'] == swin_value)].index
    
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
    assigned_values = np.random.choice(service_values, size=len(specific_rows_index), p=weights)
    
    # Assign the selected values to the DataFrame using .loc
    df2.loc[specific_rows_index, 'service'] = assigned_values
    
combinations_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_swin_with_hyphen = combinations_swin_with_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_swin_with_hyphen)
if df2[df2['service']=='-'].value_counts().sum()==0:
    print('We see that there are about',combinations_swin_with_hyphen[combinations_swin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which means we are done with our preprocessing")
else:
    print('We see that there are about',combinations_swin_with_hyphen[combinations_swin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less, thus we would drop them")
    df2 = df2[df2['service'] != '-']

#%%
#assigning preprocessed dataset to a variable
df_percentage_prep_correlation=df2.copy()
df_percentage_prep_correlation['label'].value_counts()


#%%
# #checking for distribution of service values in each type 1 and type 2
# print(df_random_prep_correlation['service'].value_counts())
# print(df_weight_prep_correlation['service'].value_counts())


#%%%%
'''
#testinggg

'''

#%%



'''
issues- the order of us applying the preprocessing to dwin values and service values would make a difference in the answer, 
howerver it was done with proto as 1st since its most correlated comparatively
'''

#%%
#Method 2: Establishing pre-processing with Most correlated Columns 

#%%
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_testing-set.csv",index_col=0)

df2 = df.copy()
# label_encoding all the object columns
from sklearn.preprocessing import LabelEncoder
# Create a dictionary to store mappings
label_mappings = {} 
# Iterate over columns
for column in df.columns:
    # Check if column has object dtype
    if df2[column].dtype == 'object':
        # Apply label encoding
        label_encoder = LabelEncoder()
        df2[column] = label_encoder.fit_transform(df2[column])
        # Store mappings
        label_mappings[column] = {index: label for index, label in enumerate(label_encoder.classes_)}
        
#getting top two highly correlated columns
# Calculate correlation matrixm
correlation_matrix = df2.corr()


'''
checking for correlation between service and all other columns
'''
service_correlation = df2.corr()['service']

print('top 5 correlated columns with service:\n',service_correlation.drop('service').sort_values(ascending=False).head(5))

#theres a considerable difference between the ----1st, 2nd---- and----the rest of the columns
#%%
'''
TESTINGGGG
'''


#%%
#Type 2: Weighted Assigning of values 
#%%

df2 = df.copy()
# Filter DataFrame to include only rows where 'service' is '-'

combinations_ct_flw_http_mthd_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_ct_flw_http_mthd_trans_depth_with_hyphen = combinations_ct_flw_http_mthd_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_ct_flw_http_mthd_trans_depth_with_hyphen)

unique_service_values_dict = {}

# Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
for index, row in combinations_ct_flw_http_mthd_trans_depth_with_hyphen.iterrows():
    # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
    current_ct_flw_http_mthd = row['ct_flw_http_mthd']
    current_trans_depth = row['trans_depth']

    # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
    filtered_rows = df2[(df2['ct_flw_http_mthd'] == current_ct_flw_http_mthd) & (df2['trans_depth'] == current_trans_depth) & (df2['service'] != '-')]

    # Extract unique values in the 'service' column for the filtered 
    unique_service_values = filtered_rows['service'].unique()

    # Store the unique service values in the dictionary
    unique_service_values_dict[(current_ct_flw_http_mthd, current_trans_depth)] = set(unique_service_values)

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
    ct_flw_http_mthd_value, trans_depth_value = key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value) & (df2['trans_depth'] == trans_depth_value)]
    
    # Calculate the count of each unique service value
    service_counts = specific_rows['service'].value_counts()
    
    # Store the service counts in the dictionary
    service_counts_dict[key] = service_counts.to_dict()

# Print the dictionary containing service counts
print(service_counts_dict)
# Iterate over each key-value pair in the service counts dictionary
for key, value in service_counts_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    ct_flw_http_mthd_value, trans_depth_value = key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value) & (df2['trans_depth'] == trans_depth_value)]
    
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
        
    # Assign the selected values to the DataFrame using .loc[]
    df2.loc[specific_rows.index, 'service'] = assigned_values

    
    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows

for key, value in service_counts_dict.items():
    # Get the ct_flw_http_mthd value from the key
    ct_flw_http_mthd_value, _ = key
    
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' matches the current key
    specific_rows_index = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value)].index
    
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
    assigned_values = np.random.choice(service_values, size=len(specific_rows_index), p=weights)
    
    # Assign the selected values to the DataFrame using .loc
    df2.loc[specific_rows_index, 'service'] = assigned_values

#checking for values remaining after preprocessing
combinations_ct_flw_http_mthd_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_ct_flw_http_mthd_trans_depth_with_hyphen = combinations_ct_flw_http_mthd_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_ct_flw_http_mthd_trans_depth_with_hyphen)

print('We see that there are about',combinations_ct_flw_http_mthd_trans_depth_with_hyphen[combinations_ct_flw_http_mthd_trans_depth_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which means we are done replacing values for -")
# # Drop the rows where the 'service' column contains '-'
# df2 = df2[df2['service'] != '-']
# #%%
# combinations_ct_flw_http_mthd_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_ct_flw_http_mthd_with_hyphen = combinations_ct_flw_http_mthd_with_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_ct_flw_http_mthd_with_hyphen)
# print('We see that there are about',combinations_ct_flw_http_mthd_with_hyphen[combinations_ct_flw_http_mthd_trans_depth_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")


# unique_service_values_dict = {}

# # Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
# for index, row in combinations_ct_flw_http_mthd_with_hyphen.iterrows():
#     # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
#     current_ct_flw_http_mthd = row['ct_flw_http_mthd']

#     # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
#     filtered_rows = df2[(df2['ct_flw_http_mthd'] == current_ct_flw_http_mthd) & (df2['service'] != '-')]

#     # Extract unique values in the 'service' column for the filtered rows
#     unique_service_values = filtered_rows['service'].unique()

#     # Store the unique service values in the dictionary
#     unique_service_values_dict[(current_ct_flw_http_mthd)] = set(unique_service_values)

# # Print the dictionary
# print(unique_service_values_dict)

# # Create a new dictionary to store non-empty values
# non_empty_service_values_dict = {}

# # Iterate over the items in the original dictionary
# for key, value in unique_service_values_dict.items():
#     # Check if the set of unique service values is not empty
#     if value:
#         # Add the key-value pair to the new dictionary
#         non_empty_service_values_dict[key] = value


# service_counts_dict = {}

# # Iterate over each key-value pair in the non-empty service values dictionary
# for key, value in non_empty_service_values_dict.items():
#     # Get the ct_flw_http_mthd and trans_depth values from the key
#     ct_flw_http_mthd_value= key
    
#     # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
#     specific_rows = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value)]
    
#     # Calculate the count of each unique service value
#     service_counts = specific_rows['service'].value_counts()
    
#     # Store the service counts in the dictionary
#     service_counts_dict[key] = service_counts.to_dict()
# # Print the dictionary containing service counts
# print(service_counts_dict)


# # Iterate over each key-value pair in the service counts dictionary
# for key, value in service_counts_dict.items():
#     ct_flw_http_mthd_value= key
#     specific_rows = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value)]
    
#     # Get the count of '-' if it exists
#     count_dash = value.get('-', 0)
    
#     # Remove '-' from the counts dictionary
#     if '-' in value:
#         del value['-']
    
#     # Calculate the total count of service values excluding '-'
#     total_count = sum(value.values())
    
#     # Calculate the weights and corresponding service values
#     weights = [count / total_count for count in value.values()]
#     service_values = list(value.keys())
    
#     # Randomly assign the service values based on the calculated weights
#     assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
    
#     # Assign the selected values to the DataFrame
#     specific_rows['service'] = assigned_values
    
#     # Update the specific rows back into the main DataFrame df2
#     df2.loc[specific_rows.index] = specific_rows
    
# combinations_ct_flw_http_mthd_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_ct_flw_http_mthd_with_hyphen = combinations_ct_flw_http_mthd_with_hyphen.sort_values(by='count', ascending=False)
# # Check the stored values
# print(combinations_ct_flw_http_mthd_with_hyphen)
# print('We see that there are about',combinations_ct_flw_http_mthd_with_hyphen[combinations_ct_flw_http_mthd_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")





# #%%
# combinations_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_trans_depth_with_hyphen = combinations_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_trans_depth_with_hyphen)

# unique_service_values_dict = {}

# # Iterate over each unique combination of 'ct_flw_http_mthd' and 'trans_depth'
# for index, row in combinations_trans_depth_with_hyphen.iterrows():
#     # Get the values of 'ct_flw_http_mthd' and 'trans_depth' for the current row
#     current_trans_depth = row['trans_depth']

#     # Filter df2 for rows matching the current combination of 'ct_flw_http_mthd' and 'trans_depth', excluding rows where 'service' is '-'
#     filtered_rows = df2[(df2['trans_depth'] == current_trans_depth) & (df2['service'] != '-')]

#     # Extract unique values in the 'service' column for the filtered rows
#     unique_service_values = filtered_rows['service'].unique()

#     # Store the unique service values in the dictionary
#     unique_service_values_dict[(current_trans_depth)] = set(unique_service_values)

# # Print the dictionary
# print(unique_service_values_dict)

# # Create a new dictionary to store non-empty values
# non_empty_service_values_dict = {}

# # Iterate over the items in the original dictionary
# for key, value in unique_service_values_dict.items():
#     # Check if the set of unique service values is not empty
#     if value:
#         # Add the key-value pair to the new dictionary
#         non_empty_service_values_dict[key] = value


# service_counts_dict = {}

# # Iterate over each key-value pair in the non-empty service values dictionary
# for key, value in non_empty_service_values_dict.items():
#     # Get the ct_flw_http_mthd and trans_depth values from the key
#     trans_depth_value= key
    
#     # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
#     specific_rows = df2[(df2['trans_depth'] == trans_depth_value)]
    
#     # Calculate the count of each unique service value
#     service_counts = specific_rows['service'].value_counts()
    
#     # Store the service counts in the dictionary
#     service_counts_dict[key] = service_counts.to_dict()
# # Print the dictionary containing service counts
# print(service_counts_dict)


# # # Iterate over each key-value pair in the service counts dictionary
# # for key, value in service_counts_dict.items():
# #     trans_depth_value= key
# #     specific_rows = df2[(df2['trans_depth'] == trans_depth_value)]
    
# #     # Get the count of '-' if it exists
# #     count_dash = value.get('-', 0)
    
# #     # Remove '-' from the counts dictionary
# #     if '-' in value:
# #         del value['-']
    
# #     # Calculate the total count of service values excluding '-'
# #     total_count = sum(value.values())
    
# #     # Calculate the weights and corresponding service values
# #     weights = [count / total_count for count in value.values()]
# #     service_values = list(value.keys())
    
# #     # Randomly assign the service values based on the calculated weights
# #     assigned_values = np.random.choice(service_values, size=len(specific_rows), p=weights)
        
# #     # Assign the selected values to the DataFrame using .loc[]
# #     df2.loc[specific_rows.index, 'service'] = assigned_values

    
# #     # Update the specific rows back into the main DataFrame df2
# #     df2.loc[specific_rows.index] = specific_rows

# for key, value in service_counts_dict.items():
#     # Get the ct_flw_http_mthd value from the key
#     trans_depth_value= key
    
#     # Filter DataFrame to include only rows where 'ct_flw_http_mthd' matches the current key
#     specific_rows_index = df2[(df2['trans_depth'] == trans_depth_value)].index
    
#     # Get the count of '-' if it exists
#     count_dash = value.get('-', 0)
    
#     # Remove '-' from the counts dictionary
#     if '-' in value:
#         del value['-']
    
#     # Calculate the total count of service values excluding '-'
#     total_count = sum(value.values())
    
#     # Calculate the weights and corresponding service values
#     weights = [count / total_count for count in value.values()]
#     service_values = list(value.keys())
    
#     # Randomly assign the service values based on the calculated weights
#     assigned_values = np.random.choice(service_values, size=len(specific_rows_index), p=weights)
    
#     # Assign the selected values to the DataFrame using .loc
#     df2.loc[specific_rows_index, 'service'] = assigned_values
    
# combinations_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_trans_depth_with_hyphen = combinations_trans_depth_with_hyphen.sort_values(by='count', ascending=False)
# # Check the stored values
# print(combinations_trans_depth_with_hyphen)
# if df2[df2['service']=='-'].value_counts().sum()==0:
#     print('We see that there are about',combinations_trans_depth_with_hyphen[combinations_trans_depth_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which means we are done with our preprocessing")
# else:
#     print('We see that there are about',combinations_trans_depth_with_hyphen[combinations_trans_depth_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less, thus we would drop them")
#     df2 = df2[df2['service'] != '-']

#%%
#assigning preprocessed dataset to a variable
df_testing_percentage_prep_correlation=df2.copy()
df_testing_percentage_prep_correlation['service'].value_counts()

#%%
#checking for distribution of service values in each type 1 and type 2
df_testing_percentage_prep_correlation['label'].value_counts()
