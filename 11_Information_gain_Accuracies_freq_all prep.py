# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:45:13 2024

@author: arsla
"""


'''
Data: random_Logically: 
'''

#%%
''' 
    Feature selection technique:  Information Gain:
        
        
        '''

import pandas as pd

# Create a DataFrame with the provided data
data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", 
              "KNeighborsClassifier", "MLPClassifier", "AdaBoostClassifier", 
              "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.8494145654180635, 0.7950247185006134, 0.7894864694165088, 
                 0.5688918039158529, 0.66875424568245, 0.8944881698489044, 
                 0.5741587718019725, 0.8725647378904922],
    "ROC_AUC": [0.9352199803372594, 0.7347222801777015, 0.78912870211604, 
                0.6193030843315741, 0.9675136592070803, 0.9696827537852862, 
                0.5043479643280446, 0.9622636947589078],
    "Mean_CV_Score": [0.9132961948742899, 0.73511314227742, 0.9068977878632557, 
                      0.8831541424339229, 0.91342758138733, 0.8802788322641399, 
                      0.7163736214636088, 0.90353651113603],    
    "Precision_class_0": [0.84, 0.87, 0.76, 0.51, 0.94, 0.91, 0.75, 0.9],
    "Recall_class_0": [0.83, 0.64, 0.79, 0.8, 0.77, 0.85, 0.01, 0.81],
    "F1_score_class_0": [0.83, 0.74, 0.77, 0.63, 0.85, 0.88, 0.02, 0.85],
    "Precision_class_1": [0.86, 0.76, 0.82, 0.7, 0.84, 0.89, 0.55, 0.86],
    "Recall_class_1": [0.87, 0.93, 0.79, 0.38, 0.96, 0.96, 1.0, 0.92],
    "F1_score_class_1": [0.86, 0.83, 0.81, 0.49, 0.89, 0.91, 0.71, 0.89],
    "fnr": [0.13087884937792288,0.07483068234463588, 0.20711638577605224, 
            0.6233124503661872, 0.042423172803282665, 0.070987381981823, 
            0.0033530397952880966, 0.07734051001500045],

}

freq_Random_logically_information_gain_accuracies = pd.DataFrame(data)



#%%


'''
Data: Random_Correlation: 
'''

#%%
''' 
    Feature selection technique:  Information Gain:
        
        
        '''
import pandas as pd

# Create a DataFrame with the provided data
data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", 
              "KNeighborsClassifier", "MLPClassifier", "AdaBoostClassifier", 
              "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.8935650779769713, 0.7169897682553564, 0.8709736190059758, 
                 0.7857941019287762, 0.7470485351989505, 0.813280377010154, 
                 0.7195974493514065, 0.8301510955643007],
    "ROC_AUC": [0.9777352279041595, 0.8176786906689626, 0.8630658934324776, 
                0.8819673624144748, 0.7346629965467983, 0.9379091805561848, 
                0.8003906413582911, 0.9564379136151063],
    "Mean_CV_Score": [0.9153739811482341, 0.7555649017219258, 0.902196993078493, 
                      0.8286770377565285, 0.7855075771670219, 0.887714735164576, 
                      0.7507116773049842, 0.9031315138064651],
    "Precision_class_0": [0.96, 0.78, 0.92, 0.81, 0.78, 0.97, 0.71, 0.96],
    "Recall_class_0": [0.8, 0.51, 0.78, 0.68, 0.61, 0.6, 0.63, 0.65],
    "F1_score_class_0": [0.87, 0.62, 0.84, 0.74, 0.68, 0.74, 0.67, 0.77],
    "Precision_class_1": [0.85, 0.69, 0.84, 0.77, 0.73, 0.75, 0.72, 0.77],
    "Recall_class_1": [0.97, 0.88, 0.94, 0.87, 0.86, 0.99, 0.79, 0.98],
    "F1_score_class_1": [0.91, 0.78, 0.89, 0.82, 0.79, 0.85, 0.75, 0.86],
    "fnr": [0.027861113562163593, 0.11572399188211419, 0.055325156622253596, 
            0.1303053030971499, 0.13985705461925352, 0.014647489632048002, 
            0.2105356039883526, 0.020603547163151856],

}

freq_Random_Correlation_information_gain_accuracies = pd.DataFrame(data)

# Save the DataFrame to a CSV file
freq_Random_Correlation_information_gain_accuracies.to_csv("freq_Random_Correlation_information_gain_accuracies.csv", index=False)

#%%


#%%
'''
Data: Percentage_Logically: 
'''


#%%
''' 
    Feature selection technique:  Information Gain:
        
        
        '''



import pandas as pd

# Define the data as dictionaries
data = {
    'Model': ['AdaBoostClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'LogisticRegression', 'MLPClassifier', 'RandomForestClassifier', 'eXtreme Gradient Boosting (XGBoost)'],
    'Accuracy': [0.892659759252736, 0.8731400391123204, 0.7562889453034849, 0.8014867540418089, 0.8780230058182613, 0.8909349302173042, 0.8935950538705406],
    'ROC_AUC': [0.973763244416773, 0.8656154985024582, 0.8452873463621502, 0.9083433095976154, 0.9591200123076604, 0.9788538090158964, 0.9814410126321633],
    'Mean_CV_Score': [0.9173840590016139, 0.9172842770230819, 0.7878032242585482, 0.861643115389018, 0.9179883653950712, 0.9311813358173543, 0.9285614547497125],
    'Precision_class_0': [0.92, 0.93, 0.68, 0.87, 0.92, 0.96, 0.95],
    'Recall_class_0': [0.84, 0.77, 0.86, 0.65, 0.8, 0.79, 0.81],
    'F1_score_class_0': [0.88, 0.85, 0.76, 0.75, 0.86, 0.87, 0.87],
    'Precision_class_1': [0.88, 0.84, 0.86, 0.76, 0.85, 0.85, 0.86],
    'Recall_class_1': [0.94, 0.95, 0.67, 0.92, 0.94, 0.97, 0.96],
    'F1_score_class_1': [0.91, 0.89, 0.75, 0.84, 0.89, 0.91, 0.91],
    'fnr': [0.06258686492091156, 0.04621765315802246, 0.33192878731055175, 0.07606609455315581, 0.058792384566171765, 0.026406935957113548, 0.03529749167199806]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display DataFrame
print(df)

# New data
new_data = {
    "Model": ["KNeighborsClassifier"],
    "Accuracy": [0.8400281803053676],
    "ROC_AUC": [0.9114274900045427],
    "Mean_CV_Score": [0.8993841402786741],
    "Precision_class_0": [0.85],
    "Recall_class_0": [0.78],
    "F1_score_class_0": [0.81],
    "Precision_class_1": [0.83],
    "Recall_class_1": [0.89],
    "F1_score_class_1": [0.86],
    "fnr": [0.10805444638090406]
}

# Append new data to the existing DataFrame
freq_Percentage_Logically_information_gain_accuracies = pd.concat([pd.DataFrame(new_data),df], ignore_index=True)

# Display the updated DataFrame
print(freq_Percentage_Logically_information_gain_accuracies)
#%%


'''
Data: Percentage_Correlation: 
'''

#%%

''' 
    Feature selection technique:  Information Gain:
        
        
        '''
import pandas as pd

data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", "MLPClassifier", "AdaBoostClassifier", "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.890485502933424, 0.816185212627692, 0.8663864831707703, 0.865596948753143, 0.8895623549989675, 0.8036531150169447, 0.8967046048076572],
    "ROC_AUC": [0.9785858967084635, 0.8767345056215492, 0.8594949403504005, 0.9534718343602953, 0.9700878798591512, 0.8440709391834976, 0.9814654664105933],
    "Mean_CV_Score": [0.926888493259657, 0.853188341705841, 0.913723539840632, 0.9119977616540265, 0.9118513926351144, 0.7850867386359263, 0.9207265849888016],
    "Precision_class_0": [0.96, 0.84, 0.92, 0.88, 0.9, 0.68, 0.95],
    "Recall_class_0": [0.79, 0.62, 0.77, 0.81, 0.85, 0.87, 0.81],
    "F1_score_class_0": [0.87, 0.71, 0.84, 0.84, 0.87, 0.76, 0.88],
    "Precision_class_1": [0.85, 0.74, 0.84, 0.86, 0.88, 0.86, 0.86],
    "Recall_class_1": [0.97, 0.9, 0.94, 0.91, 0.93, 0.66, 0.96],
    "F1_score_class_1": [0.91, 0.82, 0.89, 0.88, 0.9, 0.75, 0.91],
    "fnr": [0.02631869222793355, 0.09521498378521476, 0.058196739394206796, 0.09122195503981999, 0.07458801208939089, 0.3376646297072514, 0.036135807099208016]
}



freq_Percentage_Correlation_info_gain_accuracies = pd.DataFrame(data)

# New data
new_data = {
    "Model": ["KNeighborsClassifier"],
    "Accuracy": [0.8390078589041262],
    "ROC_AUC": [0.9107862793894945],
    "Mean_CV_Score": [0.8990912981576893],
    "Precision_class_0": [0.86],
    "Recall_class_0": [0.77],
    "F1_score_class_0": [0.81],
    "Precision_class_1": [0.83],
    "Recall_class_1": [0.89],
    "F1_score_class_1": [0.86],
    "fnr": [0.10518652518255421]
}

# Convert new data to DataFrame
new_df = pd.DataFrame(new_data)

# Combine with existing DataFrame
freq_Percentage_Correlation_info_gain_accuracies = pd.concat([freq_Percentage_Correlation_info_gain_accuracies, new_df], ignore_index=True)

# Display the combined DataFrame
print(freq_Percentage_Correlation_info_gain_accuracies)


# #%%
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define a custom color palette
# custom_palette = sns.color_palette("viridis", n_colors=len(freq_Random_logically_information_gain_accuracies))

# # Set style for seaborn plots
# sns.set(style="whitegrid")

# # Reorder the datasets to have the same order of model names
# freq_Random_logically_information_gain_accuracies = freq_Random_logically_information_gain_accuracies.sort_values(by="Model")
# freq_Random_Correlation_information_gain_accuracies = freq_Random_Correlation_information_gain_accuracies.sort_values(by="Model")
# freq_Percentage_Logically_information_gain_accuracies = freq_Percentage_Logically_information_gain_accuracies.sort_values(by="Model")
# freq_Percentage_Correlation_info_gain_accuracies = freq_Percentage_Correlation_info_gain_accuracies.sort_values(by="Model")

# # Plot accuracy for each model with Information Gain feature selection for random_Logically dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="Accuracy", data=freq_Random_logically_information_gain_accuracies, palette=custom_palette)
# plt.title("Accuracy of Models with Information Gain Feature Selection (random_Logically)")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Plot accuracy for each model with Information Gain feature selection for Random_Correlation dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="Accuracy", data=freq_Random_Correlation_information_gain_accuracies, palette=custom_palette)
# plt.title("Accuracy of Models with Information Gain Feature Selection (Random_Correlation)")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Plot accuracy for each model with Information Gain feature selection for Percentage_Logically dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="Accuracy", data=freq_Percentage_Logically_information_gain_accuracies, palette=custom_palette)
# plt.title("Accuracy of Models with Information Gain Feature Selection (Percentage_Logically)")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Plot accuracy for each model with Information Gain feature selection for Percentage_Correlation dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="Accuracy", data=freq_Percentage_Correlation_info_gain_accuracies, palette=custom_palette)
# plt.title("Accuracy of Models with Information Gain Feature Selection (Percentage_Correlation)")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # Find the maximum accuracy for each case
# max_accuracy_random_logically = freq_Random_logically_information_gain_accuracies['Accuracy'].max()
# max_accuracy_random_corr = freq_Random_Correlation_information_gain_accuracies['Accuracy'].max()
# max_accuracy_percentage_logically = freq_Percentage_Logically_information_gain_accuracies['Accuracy'].max()
# max_accuracy_percentage_corr = freq_Percentage_Correlation_info_gain_accuracies['Accuracy'].max()

# # Find the model with the maximum accuracy for each case
# max_accuracy_model_random_logically = freq_Random_logically_information_gain_accuracies.loc[freq_Random_logically_information_gain_accuracies['Accuracy'].idxmax()]['Model']
# max_accuracy_model_random_corr = freq_Random_Correlation_information_gain_accuracies.loc[freq_Random_Correlation_information_gain_accuracies['Accuracy'].idxmax()]['Model']
# max_accuracy_model_percentage_logically = freq_Percentage_Logically_information_gain_accuracies.loc[freq_Percentage_Logically_information_gain_accuracies['Accuracy'].idxmax()]['Model']
# max_accuracy_model_percentage_corr = freq_Percentage_Correlation_info_gain_accuracies.loc[freq_Percentage_Correlation_info_gain_accuracies['Accuracy'].idxmax()]['Model']

# # Print the model with the maximum accuracy for each case
# print("Model with Max Accuracy for random_Logically:", max_accuracy_model_random_logically)
# print("Model with Max Accuracy for Random_Correlation:", max_accuracy_model_random_corr)
# print("Model with Max Accuracy for Percentage_Logically:", max_accuracy_model_percentage_logically)
# print("Model with Max Accuracy for Percentage_Correlation:", max_accuracy_model_percentage_corr)



# # Find the minimum accuracy for each case
# min_accuracy_random_logically = freq_Random_logically_information_gain_accuracies['Accuracy'].min()
# min_accuracy_random_corr = freq_Random_Correlation_information_gain_accuracies['Accuracy'].min()
# min_accuracy_percentage_logically = freq_Percentage_Logically_information_gain_accuracies['Accuracy'].min()
# min_accuracy_percentage_corr = freq_Percentage_Correlation_info_gain_accuracies['Accuracy'].min()

# # Find the model with the minimum accuracy for each case
# min_accuracy_model_random_logically = freq_Random_logically_information_gain_accuracies.loc[freq_Random_logically_information_gain_accuracies['Accuracy'].idxmin()]['Model']
# min_accuracy_model_random_corr = freq_Random_Correlation_information_gain_accuracies.loc[freq_Random_Correlation_information_gain_accuracies['Accuracy'].idxmin()]['Model']
# min_accuracy_model_percentage_logically = freq_Percentage_Logically_information_gain_accuracies.loc[freq_Percentage_Logically_information_gain_accuracies['Accuracy'].idxmin()]['Model']
# min_accuracy_model_percentage_corr = freq_Percentage_Correlation_info_gain_accuracies.loc[freq_Percentage_Correlation_info_gain_accuracies['Accuracy'].idxmin()]['Model']

# # Print the model with the minimum accuracy for each case
# print("Model with Min Accuracy for random_Logically:", min_accuracy_model_random_logically)
# print("Model with Min Accuracy for Random_Correlation:", min_accuracy_model_random_corr)
# print("Model with Min Accuracy for Percentage_Logically:", min_accuracy_model_percentage_logically)
# print("Model with Min Accuracy for Percentage_Correlation:", min_accuracy_model_percentage_corr)

# # Print the minimum accuracy for each case
# print("Min Accuracy for random_Logically:", min_accuracy_random_logically)
# print("Min Accuracy for Random_Correlation:", min_accuracy_random_corr)
# print("Min Accuracy for Percentage_Logically:", min_accuracy_percentage_logically)
# print("Min Accuracy for Percentage_Correlation:", min_accuracy_percentage_corr)

# #%%%

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define a custom color palette
# custom_palette = sns.color_palette("viridis", n_colors=len(freq_Random_logically_information_gain_accuracies))

# # Sort dataframes by the "Model" column
# freq_Random_logically_information_gain_accuracies = freq_Random_logically_information_gain_accuracies.sort_values(by="Model")
# freq_Random_Correlation_information_gain_accuracies = freq_Random_Correlation_information_gain_accuracies.sort_values(by="Model")
# freq_Percentage_Logically_information_gain_accuracies = freq_Percentage_Logically_information_gain_accuracies.sort_values(by="Model")
# freq_Percentage_Correlation_info_gain_accuracies = freq_Percentage_Correlation_info_gain_accuracies.sort_values(by="Model")

# # Plot FNR for each model with Information Gain feature selection for random_Logically dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="fnr", data=freq_Random_logically_information_gain_accuracies, palette=custom_palette)
# plt.title("False Negative Rate (FNR) of Models with Information Gain Feature Selection (random_Logically)")
# plt.xticks(rotation=45, ha='right')
# plt.ylim(0, 1)  # Set common y-axis range up to 1
# plt.tight_layout()
# plt.show()

# # Plot FNR for each model with Information Gain feature selection for Random_Correlation dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="fnr", data=freq_Random_Correlation_information_gain_accuracies, palette=custom_palette)
# plt.title("False Negative Rate (FNR) of Models with Information Gain Feature Selection (Random_Correlation)")
# plt.xticks(rotation=45, ha='right')
# plt.ylim(0, 1)  # Set common y-axis range up to 1
# plt.tight_layout()
# plt.show()

# # Plot FNR for each model with Information Gain feature selection for Percentage_Logically dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="fnr", data=freq_Percentage_Logically_information_gain_accuracies, palette=custom_palette)
# plt.title("False Negative Rate (FNR) of Models with Information Gain Feature Selection (Percentage_Logically)")
# plt.xticks(rotation=45, ha='right')
# plt.ylim(0, 1)  # Set common y-axis range up to 1
# plt.tight_layout()
# plt.show()

# # Plot FNR for each model with Information Gain feature selection for Percentage_Correlation dataset
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Model", y="fnr", data=freq_Percentage_Correlation_info_gain_accuracies, palette=custom_palette)
# plt.title("False Negative Rate (FNR) of Models with Information Gain Feature Selection (Percentage_Correlation)")
# plt.xticks(rotation=45, ha='right')
# plt.ylim(0, 1)  # Set common y-axis range up to 1
# plt.tight_layout()
# plt.show()


#%%


# Extract model names and accuracy columns from each DataFrame
import pandas as pd

accuracy_random_logically = freq_Random_logically_information_gain_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_random_logically'})
accuracy_percentage_logically = freq_Percentage_Logically_information_gain_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_percentage_logically'})
accuracy_random_corr = freq_Random_Correlation_information_gain_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_random_corr'})
accuracy_percentage_corr = freq_Percentage_Correlation_info_gain_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_percentage_corr'})

# Concatenate model names and accuracy columns into one DataFrame
combined_accuracy_df_infogain = pd.concat([accuracy_random_logically.set_index('Model'), 
                                  accuracy_percentage_logically.set_index('Model'), 
                                  accuracy_random_corr.set_index('Model'), 
                                  accuracy_percentage_corr.set_index('Model')], axis=1).sort_values('Model')

#%%



model_abbreviations = {
    'eXtreme Gradient Boosting (XGBoost)': 'XGB',
    'AdaBoostClassifier': 'ADB',
    'RandomForestClassifier': 'RF',
    'MLPClassifier': 'MLP',
    'DecisionTreeClassifier': 'DT',
    'KNeighborsClassifier': 'KNN',
    'LogisticRegression': 'LR',
    'GaussianNB': 'GNB'
}

# Replace the long model names with abbreviations
combined_accuracy_df_infogain.index = combined_accuracy_df_infogain.index.map(model_abbreviations)

# Print the DataFrame to check if the replacement is successful
print(combined_accuracy_df_infogain)

combined_accuracy_df_infogain.index
# #%%
# import matplotlib.pyplot as plt

# # Define a function to create a set of 4 graphs
# def create_graphs(dataframe):
#     # Get the columns of the DataFrame excluding the 'Model' column
#     columns = dataframe.columns
    
#     # Calculate the number of rows required for subplots
#     num_rows = (len(columns) + 1) // 2
    
#     # Create subplots
#     fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))
    
#     # Flatten the axes array for easier iteration
#     axes = axes.flatten()
    
#     # Plot each column
#     for i, column in enumerate(columns):
#         ax = axes[i]
#         dataframe.plot(kind='bar', y=column, ax=ax, legend=False)
#         ax.set_title(column)
#         ax.set_ylabel('Accuracy')
#         ax.set_xlabel('Model')
#         ax.grid(True)
#         ax.tick_params(axis='x', rotation=45)  # Adjust rotation for x-axis labels
#         ax.tick_params(axis='y', rotation=0)   # Adjust rotation for y-axis labels
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Set rotation to 0 degrees
#         # Adjust layout
#         plt.tight_layout()

#     # Set title over the overall plot
#     fig.suptitle('Information Gain', fontsize=16)
    
#     # Show plots
#     plt.show()

# # Create sets of 4 graphs
# for i in range(0, len(combined_accuracy_df_infogain.columns), 4):
#     create_graphs(combined_accuracy_df_infogain.iloc[:, i:i+4])
#     columns = combined_accuracy_df_infogain.columns
#     print("Columns in this set of graphs:", combined_accuracy_df_infogain.index)
# #%%

# import matplotlib.pyplot as plt

# # Set the 'Model' column as the index for easier plotting
# df.set_index('Model', inplace=True)

# def create_graphs(dataframe):
#     # Get the columns of the DataFrame excluding the 'Model' column
#     columns = dataframe.columns
    
#     # Calculate the number of rows required for subplots
#     num_rows = (len(columns) + 1) // 2
    
#     # Create subplots
#     fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))
    
#     # Flatten the axes array for easier iteration
#     axes = axes.flatten()
    
#     # Plot each column
#     for i, column in enumerate(columns):
#         ax = axes[i]
#         dataframe[column].plot(kind='bar', ax=ax)
#         ax.set_title(column)
#         ax.set_ylabel('Accuracy')
#         ax.set_xlabel('Model')
#         ax.grid(True)
#         plt.xticks(rotation=45, ha='right')
#         ax.set_xticklabels(dataframe.index, rotation=0)  # Use provided index for model names
        
#     # Set title over the overall plot
#     fig.suptitle('Gini Index', fontsize=16)
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Show plots
#     plt.show()

# # Create sets of 4 graphs
# for i in range(0, len(combined_accuracy_df_infogain.columns), 4):
#     create_graphs(combined_accuracy_df_infogain.iloc[:, i:i+4])
# #%%
#%%
import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrame for the second set of data
data2 = {
    'Model': ['ADB', 'DT', 'GNB', 'KNN', 'LR', 'MLP', 'RF', 'XGB'],
    'Randomly Relation based': [0.8944881698489044, 0.7894864694165088, 0.5741587718019725, 0.5688918039158529, 0.7950247185006134, 0.66875424568245, 0.8494145654180635, 0.8725647378904922],
    'Most Frequent Relation': [0.892659759252736, 0.8731400391123204, 0.7562889453034849, 0.8400281803053676, 0.8014867540418089, 0.8780230058182613, 0.8909349302173042, 0.8935950538705406],
    'Random Correlated': [0.813280377010154, 0.8709736190059758, 0.7195974493514065, 0.7857941019287762, 0.7169897682553564, 0.7470485351989505, 0.8935650779769713, 0.8301510955643007],
    'Most Frequent Correlated': [0.8895623549989675, 0.8663864831707703, 0.8036531150169447, 0.8390078589041262, 0.816185212627692, 0.865596948753143, 0.890485502933424, 0.8967046048076572]
}

df2 = pd.DataFrame(data2)

# Set the 'Model' column as the index for easier plotting
df2.set_index('Model', inplace=True)

# Define a function to create a set of 4 graphs
def create_graphs(dataframe):
    # Get the columns of the DataFrame excluding the 'Model' column
    columns = dataframe.columns
    
    # Calculate the number of rows required for subplots
    num_rows = (len(columns) + 1) // 2
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Plot each column
    for i, column in enumerate(columns):
        ax = axes[i]
        dataframe[column].plot(kind='bar', ax=ax)
        ax.set_title(column)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Model')
        ax.set_ylim(0, 1)  # Set y-axis limits
        ax.grid(True)
        plt.xticks(rotation=45, ha='right')
        ax.set_xticklabels(dataframe.index, rotation=0)  # Use provided index for model names
    
    # Set title over the overall plot
    fig.suptitle('Information Gain', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plots
    plt.show()

# Create sets of 4 graphs for the second set of data
for i in range(0, len(df2.columns), 4):
    create_graphs(df2.iloc[:, i:i+4])
