# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:41:22 2024

@author: arsla
"""

#%%
#Method 2: Establishing pre-processing with Most correlated Columns 

#%%
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_training-set.csv", index_col=0)

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


#Type 1: Random Assigning of values 
#%%


# Filter DataFrame to include only rows where 'service' is '-'
df2 = df.copy()
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

    # Extract unique values in the 'service' column for the filtered rows
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

# Print the filtered dictionary
print(non_empty_service_values_dict)

'''
treating for dwin,swin: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the dwin and swin values from the key
    dwin_value, swin_value = key

    # Filter DataFrame to include only rows where 'dwin' and 'swin' match the current key
    # Filter DataFrame to include only rows where 'dwin' and 'swin' match the current key
    specific_rows = df2[(df2['dwin'] == dwin_value) & (df2['swin'] == swin_value)].copy()


    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2 using .loc[]
    df2.loc[specific_rows.index, 'service'] = specific_rows['service']
# Verify if the rows have been updated in df2


print(df2.groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False))

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_dwin_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_swin_with_hyphen = combinations_dwin_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_swin_with_hyphen)


#%%
# Filter DataFrame to include only rows where 'service' is '-'

combinations_dwin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin','service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_with_hyphen = combinations_dwin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_with_hyphen)

print('We see that there are about',combinations_dwin_swin_with_hyphen[combinations_dwin_swin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

'''
treating for dwin,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the dwin and swin values from the key
    dwin_value = key

    # Filter DataFrame to include only rows where 'dwin' match the current key
    specific_rows = df2[(df2['dwin'] == dwin_value)].copy()

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))
    # Update the specific rows back into the main DataFrame df2 using .loc[]
    df2.loc[specific_rows.index, 'service'] = specific_rows['service']

print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_dwin_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_swin_with_hyphen = combinations_dwin_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_swin_with_hyphen)





#%%
# Filter DataFrame to include only rows where 'service' is '-'

combinations_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['swin','service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_swin_with_hyphen = combinations_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_swin_with_hyphen)

print('We see that there are about',combinations_dwin_swin_with_hyphen[combinations_dwin_swin_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

'''
treating for swin,swin: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the swin and swin values from the key
    swin_value = key

    # Filter DataFrame to include only rows where 'swin' match the current key
    specific_rows = df2[(df2['swin'] == swin_value)].copy()

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    
combinations_dwin_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'})
print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")
#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_dwin_swin_with_hyphen = df2[df2['service'] == '-'].groupby(['dwin', 'swin', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_dwin_swin_with_hyphen = combinations_dwin_swin_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_dwin_swin_with_hyphen)

print('We see that there are about',df2[df2['service']=='-'].value_counts().sum()," '-' service values\nthus we have fully pre processed the data for the missing '-' values in service column")
#%%
df_random_prep_correlation=df2.copy()



#%%%
'''
TESTING
'''

#%%

#%%
#Method 2: Establishing pre-processing with Most correlated Columns 

#%%
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_testing-set.csv", index_col=0)

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


#Type 1: Random Assigning of values 
#%%


# Filter DataFrame to include only rows where 'service' is '-'
df2 = df.copy()
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

    # Extract unique values in the 'service' column for the filtered rows
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

# Print the filtered dictionary
print(non_empty_service_values_dict)

'''
treating for ct_flw_http_mthd,trans_depth: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the ct_flw_http_mthd and trans_depth values from the key
    ct_flw_http_mthd_value, trans_depth_value = key

    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    # Filter DataFrame to include only rows where 'ct_flw_http_mthd' and 'trans_depth' match the current key
    specific_rows = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value) & (df2['trans_depth'] == trans_depth_value)].copy()


    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2 using .loc[]
    df2.loc[specific_rows.index, 'service'] = specific_rows['service']
# Verify if the rows have been updated in df2


print(df2.groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False))

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_ct_flw_http_mthd_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_ct_flw_http_mthd_trans_depth_with_hyphen = combinations_ct_flw_http_mthd_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_ct_flw_http_mthd_trans_depth_with_hyphen)


# #%%
# # Filter DataFrame to include only rows where 'service' is '-'

# combinations_ct_flw_http_mthd_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd','service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_ct_flw_http_mthd_with_hyphen = combinations_ct_flw_http_mthd_with_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_ct_flw_http_mthd_with_hyphen)

# print('We see that there are about',combinations_ct_flw_http_mthd_trans_depth_with_hyphen[combinations_ct_flw_http_mthd_trans_depth_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

# '''
# treating for ct_flw_http_mthd,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
# '''


# # Iterate over each key-value pair in the dictionary
# for key, value in non_empty_service_values_dict.items():
#     # Get the ct_flw_http_mthd and trans_depth values from the key
#     ct_flw_http_mthd_value = key

#     # Filter DataFrame to include only rows where 'ct_flw_http_mthd' match the current key
#     specific_rows = df2[(df2['ct_flw_http_mthd'] == ct_flw_http_mthd_value)].copy()

#     # Randomly assign the provided values to the 'service' column in specific rows
#     possible_services = list(value)  # Convert the set of unique service values to a list
#     specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))
#     # Update the specific rows back into the main DataFrame df2 using .loc[]
#     df2.loc[specific_rows.index, 'service'] = specific_rows['service']

# print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")

# #%%
# '''
# Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
# '''

# combinations_ct_flw_http_mthd_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_ct_flw_http_mthd_trans_depth_with_hyphen = combinations_ct_flw_http_mthd_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_ct_flw_http_mthd_trans_depth_with_hyphen)





# #%%
# # Filter DataFrame to include only rows where 'service' is '-'

# combinations_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['trans_depth','service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_trans_depth_with_hyphen = combinations_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_trans_depth_with_hyphen)

# print('We see that there are about',combinations_ct_flw_http_mthd_trans_depth_with_hyphen[combinations_ct_flw_http_mthd_trans_depth_with_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

# '''
# treating for trans_depth,trans_depth: replacing values randomly with '-' in service column: for values that have any value for service except '-'
# '''


# # Iterate over each key-value pair in the dictionary
# for key, value in non_empty_service_values_dict.items():
#     # Get the trans_depth and trans_depth values from the key
#     trans_depth_value = key

#     # Filter DataFrame to include only rows where 'trans_depth' match the current key
#     specific_rows = df2[(df2['trans_depth'] == trans_depth_value)].copy()

#     # Randomly assign the provided values to the 'service' column in specific rows
#     possible_services = list(value)  # Convert the set of unique service values to a list
#     specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

#     # Update the specific rows back into the main DataFrame df2
#     df2.loc[specific_rows.index] = specific_rows
    
# combinations_ct_flw_http_mthd_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
# print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")
# #%%
# '''
# Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
# '''

# combinations_ct_flw_http_mthd_trans_depth_with_hyphen = df2[df2['service'] == '-'].groupby(['ct_flw_http_mthd', 'trans_depth', 'service']).size().reset_index().rename(columns={0: 'count'})
# #descending
# combinations_ct_flw_http_mthd_trans_depth_with_hyphen = combinations_ct_flw_http_mthd_trans_depth_with_hyphen.sort_values(by='count', ascending=False)

# # Check the stored values
# print(combinations_ct_flw_http_mthd_trans_depth_with_hyphen)

# print('We see that there are about',df2[df2['service']=='-'].value_counts().sum()," '-' service values\nthus we have fully pre processed the data for the missing '-' values in service column")
#%%
df_testing_random_prep_correlation=df2.copy()

df_random_prep_logically['service'].value_counts()
df_percentage_prep_logically['service'].value_counts()
df_random_prep_correlation['service'].value_counts()
df_percentage_prep_correlation['service'].value_counts()
df_random_prep_logically['service'].value_counts()
df_testing_random_prep_correlation['service'].value_counts()




