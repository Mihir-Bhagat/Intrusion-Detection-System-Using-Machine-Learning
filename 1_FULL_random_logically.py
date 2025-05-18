# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:08:45 2024

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
#Type 1: Randomized Assigning of values

#checking for proto & State values with hyphen:
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_training-set.csv", index_col=0)
df['attack_cat'].value_counts()
df["label"].value_counts()

# f = open("outputs.txt",'w+')
# print(f.read())

# f = open("outputs.txt",'a')
# f.write(str(df['attack_cat'].value_counts()))

# f = open("outputs.txt",'r+')
# print(f.read())

df2=df.copy()
combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)

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

'''
removing all cases where the dictionary value for 'service' is empty
'''

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

'''
treating for proto,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)].copy()

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows

# Verify if the rows have been updated in df2
print(df2.groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False))

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
print(combinations_proto_state_with_state_hyphen)


#%%
# Filter DataFrame to include only rows where 'service' is '-'

combinations_proto_with_hyphen = df2[df2['service'] == '-'].groupby(['proto','service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_with_hyphen = combinations_proto_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_proto_with_hyphen)

print('We see that there are about',combinations_proto_state_with_state_hyphen[combinations_proto_state_with_state_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

'''
treating for proto,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and swin values from the key
    proto_value = key

    # Filter DataFrame to include only rows where 'proto' match the current key
    specific_rows = df2[(df2['proto'] == proto_value)]

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    

print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")
#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
print(combinations_proto_state_with_state_hyphen)
#%%
# Filter DataFrame to include only rows where 'service' is '-'

combinations_state_with_hyphen = df2[df2['service'] == '-'].groupby(['state','service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_state_with_hyphen = combinations_state_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_state_with_hyphen)

print('We see that there are about',combinations_proto_state_with_state_hyphen[combinations_proto_state_with_state_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

'''
treating for state,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the state and swin values from the key
    state_value = key

    # Filter DataFrame to include only rows where 'state' match the current key
    specific_rows = df2[(df2['state'] == state_value)].copy()

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    

print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
print(combinations_proto_state_with_state_hyphen)

#%%
print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values\nwhich makes no sense as neither their proto value (i.e. icmp) nor their individual state values(i.e ECO,PAR,URN,no) has any past history of service value other than '-'\nthus we would drop them as the total is very less compared to the whole dataset")
# Drop rows with '-' in the 'service' column
df2 = df2[df2['service'] != '-']

#%%
#storing in a variable
df_random_prep_logically=df2.copy()

#%%

'''

TESTING

'''



#%%
#Method 1: Establishing pre-processing with Most Logically Related Columns


#%%
#Type 1: Randomized Assigning of values

#checking for proto & State values with hyphen:
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\arsla\Desktop\ALL SEMESTERS\Projects\Sem 4\RIDS-\CS\Topic Ideas\Cybersecurity\Dataset\archive\UNSW_NB15_testing-set.csv", index_col=0)

df2=df.copy()
combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
# Check the stored values
print(combinations_proto_state_with_state_hyphen)

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

'''
removing all cases where the dictionary value for 'service' is empty
'''

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

'''
treating for proto,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and state values from the key
    proto_value, state_value = key

    # Filter DataFrame to include only rows where 'proto' and 'state' match the current key
    specific_rows = df2[(df2['proto'] == proto_value) & (df2['state'] == state_value)].copy()

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows

# Verify if the rows have been updated in df2
print(df2.groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'}).sort_values(by='count', ascending=False))

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
print(combinations_proto_state_with_state_hyphen)


#%%
# Filter DataFrame to include only rows where 'service' is '-'

combinations_proto_with_hyphen = df2[df2['service'] == '-'].groupby(['proto','service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_with_hyphen = combinations_proto_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_proto_with_hyphen)

print('We see that there are about',combinations_proto_state_with_state_hyphen[combinations_proto_state_with_state_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

'''
treating for proto,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the proto and swin values from the key
    proto_value = key

    # Filter DataFrame to include only rows where 'proto' match the current key
    specific_rows = df2[(df2['proto'] == proto_value)]

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    

print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")
#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
print(combinations_proto_state_with_state_hyphen)
#%%
# Filter DataFrame to include only rows where 'service' is '-'

combinations_state_with_hyphen = df2[df2['service'] == '-'].groupby(['state','service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_state_with_hyphen = combinations_state_with_hyphen.sort_values(by='count', ascending=False)

# Check the stored values
print(combinations_state_with_hyphen)

print('We see that there are about',combinations_proto_state_with_state_hyphen[combinations_proto_state_with_state_hyphen['service']=='-'].value_counts().sum()," combinations having a total of",df2[df2['service']=='-'].value_counts().sum()," '-' service which is very less compared to the dataset size we have")

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

'''
treating for state,state: replacing values randomly with '-' in service column: for values that have any value for service except '-'
'''


# Iterate over each key-value pair in the dictionary
for key, value in non_empty_service_values_dict.items():
    # Get the state and swin values from the key
    state_value = key

    # Filter DataFrame to include only rows where 'state' match the current key
    specific_rows = df2[(df2['state'] == state_value)].copy()

    # Randomly assign the provided values to the 'service' column in specific rows
    possible_services = list(value)  # Convert the set of unique service values to a list
    specific_rows['service'] = np.random.choice(possible_services, size=len(specific_rows))

    # Update the specific rows back into the main DataFrame df2
    df2.loc[specific_rows.index] = specific_rows
    

print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values")

#%%
'''
Checking for what values with '-' in service column are left out- for values that dont have any value for service except '-'
'''

combinations_proto_state_with_state_hyphen = df2[df2['service'] == '-'].groupby(['proto', 'state', 'service']).size().reset_index().rename(columns={0: 'count'})
#descending
combinations_proto_state_with_state_hyphen = combinations_proto_state_with_state_hyphen.sort_values(by='count', ascending=False)
print(combinations_proto_state_with_state_hyphen)

#%%
# print('We see that there are still about',df2[df2['service']=='-'].value_counts().sum()," '-' service values\nwhich makes no sense as neither their proto value (i.e. icmp) nor their individual state values(i.e ECO,PAR,URN,no) has any past history of service value other than '-'\nthus we would drop them as the total is very less compared to the whole dataset")
# # Drop rows with '-' in the 'service' column
df2 = df2[df2['service'] != '-']

#%%
#storing in a variable
df_testing_random_prep_logically=df2.copy()
.isnull().sum()


df_random_prep_correlation['label'].value_counts()
