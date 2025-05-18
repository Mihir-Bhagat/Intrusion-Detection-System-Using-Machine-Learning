# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:41:15 2024

@author: arsla
"""
#%%


'''
Data: Random_Logically: 
'''

#%%
''' 
    Feature selection technique:  Gini Index:
        
        
        '''

import pandas as pd

# Create a DataFrame with the provided data
data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", 
              "KNeighborsClassifier", "MLPClassifier", "AdaBoostClassifier", 
              "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.796725453043774, 0.790636972074775, 0.8387990088908323, 
                 0.5515838313171063, 0.8017221567650953, 0.8691881649905262, 
                 0.5541587718019725, 0.8535806247874459],
    "ROC_AUC": [0.9127837047870246, 0.9351068747344786, 0.8349488387178319, 
                0.5809622985731695, 0.9589250278445779, 0.9554524099675427, 
                0.5109935043796996, 0.9517363451866232],
    "Mean_CV_Score": [0.9072559248271549, 0.8784529181885492, 0.8990601046457352, 
                      0.8783039024702506, 0.8949013087838764, 0.8784348153049946, 
                      0.7793393583597622, 0.9004057694863115],
    "Precision_class_0": [0.76,0.95, 0.85, 0.5, 0.9, 0.89, 0.86, 0.87],
    "Recall_class_0": [0.81, 0.65, 0.78, 0.8, 0.78, 0.81, 0.03, 0.79],
    "F1_score_class_0": [0.78, 0.77, 0.81, 0.61, 0.84, 0.85, 0.05, 0.83],
    "Precision_class_1": [0.83, 0.77, 0.83, 0.68, 0.84, 0.86, 0.56, 0.84],
    "Recall_class_1": [0.79, 0.97, 0.89, 0.35, 0.93, 0.92, 1.0, 0.91],
    "F1_score_class_1": [0.81,0.86, 0.86, 0.46, 0.88, 0.89, 0.71, 0.87],
    "fnr": [0.21183711285626047, 0.025017097222528623, 0.10932674490426189, 
            0.6486367246095474,0.07200688301087604, 0.0824803670696197, 
            0.0033530397952880966, 0.09342186534898085]
}

freq_Random_logically_Gini_accuracies = pd.DataFrame(data)

# Save the DataFrame to a CSV file
freq_Random_logically_Gini_accuracies.to_csv("freq_Random_logically_Gini_accuracies.csv", index=False)



    #%%


'''
Data: Random_Correlation: 
'''

#%%
''' 
    Feature selection technique:  Gini Index:
        
        
        '''

import pandas as pd

# Create a DataFrame with the provided data
data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", 
              "KNeighborsClassifier", "MLPClassifier", "AdaBoostClassifier", 
              "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.8676942136714765, 0.7136957683525239, 0.8478355924792305, 
                 0.7368580867706359, 0.7473886216780838, 0.7825875722683768, 
                 0.7174974493514065, 0.7917091774765583],
    "ROC_AUC": [0.9571878712847673, 0.8103965136494475, 0.8433000571161473, 
                0.8260373049525305, 0.7259740076814659, 0.8894162184221635, 
                0.8073013657794386, 0.9376378880976626],
    "Mean_CV_Score": [0.9028667180458039, 0.7493128677112963, 0.8946241772380319, 
                      0.8132850598829755, 0.7472849166006885, 0.8786154827588876, 
                      0.7312201733623096, 0.8973903375640037],
    "Precision_class_0": [0.91, 0.79, 0.86, 0.72, 0.91, 0.94, 0.84, 0.95],
    "Recall_class_0": [0.78, 0.5, 0.78, 0.67, 0.48, 0.55, 0.39, 0.57],
    "F1_score_class_0": [0.84, 0.61, 0.82, 0.7, 0.63, 0.7, 0.53, 0.71],
    "Precision_class_1": [0.84, 0.68, 0.84, 0.75, 0.7, 0.73, 0.65, 0.73],
    "Recall_class_1": [0.94, 0.89, 0.9, 0.79, 0.96, 0.97, 0.94, 0.98],
    "F1_score_class_1": [0.89, 0.77, 0.87, 0.77, 0.81, 0.83, 0.77, 0.84],
    "fnr": [0.060641489455572226, 0.11054001588282009, 0.0998852907438454, 
            0.2114400423541869, 0.03765551927997882, 0.029471455042795375, 
            0.05978117003441278, 0.02470660901791229],
    
}

freq_Random_Correlation_Gini_accuracies = pd.DataFrame(data)

# Save the DataFrame to a CSV file
freq_Random_Correlation_Gini_accuracies.to_csv("freq_Random_Correlation_Gini_accuracies.csv", index=False)
#%%
'''
Data: Percentage_Logically: 
'''

#%%
''' 
    Feature selection technique:  Gini Index:
        
        
        '''
import pandas as pd

# Provided data
data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", "MLPClassifier", "AdaBoostClassifier", "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.8639085597677554, 0.8335661447641721, 0.8531344516379803, 0.8677104716557144, 0.8521870103368275, 0.733380300508946, 0.8687186463736077],
    "ROC_AUC": [0.9618550253032241, 0.9369724155142287, 0.8616471462436995, 0.9623265110974497, 0.9537054487095662, 0.8524718379629805, 0.9652330565173883],
    "Mean_CV_Score": [0.9155584207134513, 0.8736305869790462, 0.9050165000195793, 0.9002221826107754, 0.8869575140787639, 0.7820055332937283, 0.9147796451283428],
    "Precision_class_0": [0.92, 0.94, 0.9, 0.9, 0.91, 0.9, 0.91],
    "Recall_class_0": [0.77, 0.67, 0.75, 0.79, 0.75, 0.46, 0.78],
    "F1_score_class_0": [0.83, 0.78, 0.82, 0.84, 0.82, 0.61, 0.84],
    "Precision_class_1": [0.83, 0.78, 0.82, 0.85, 0.82, 0.68, 0.84],
    "Recall_class_1": [0.94, 0.97, 0.93, 0.93, 0.94, 0.96, 0.94],
    "F1_score_class_1": [0.88, 0.86, 0.88, 0.89, 0.87, 0.8, 0.89],
    "fnr": [0.05625537735224691, 0.034547419973968096, 0.06622691874958636, 0.06993315537514615, 0.06287365704074654, 0.04330561009508262, 0.06121908711862163],

}

# Create DataFrame
PART_1 = pd.DataFrame(data)

# Provided data
new_data = {
    "Model": ["KNeighborsClassifier"],
    "Accuracy": [0.8387892186038602],
    "ROC_AUC": [0.9149057554146021],
    "Mean_CV_Score": [0.8930041767674644],
    "Precision_class_0": [0.87],
    "Recall_class_0": [0.76],
    "F1_score_class_0": [0.81],
    "Precision_class_1": [0.82],
    "Recall_class_1": [0.9],
    "F1_score_class_1": [0.86],
    "fnr": [0.09627390853537471],

}

# Convert new data to DataFrame
new_df = pd.DataFrame(new_data)

# Combine with existing DataFrame
freq_Percentage_Logically_Gini_accuracies = pd.concat([PART_1, new_df], ignore_index=True)

# Display the combined DataFrame
print(freq_Percentage_Logically_Gini_accuracies)
#%%


'''
Data: Percentage_Correlation: 
'''

#%%
''' 
    Feature selection technique:  Gini Index:
        
        
        '''

import pandas as pd

data = {
    "Model": ["RandomForestClassifier", "LogisticRegression", "DecisionTreeClassifier", "MLPClassifier", "AdaBoostClassifier", "GaussianNB", "eXtreme Gradient Boosting (XGBoost)"],
    "Accuracy": [0.861685716715051, 0.821820301966548, 0.8527093177207964, 0.8597665407460493, 0.8634105457504828, 0.7289953478202776, 0.8702248351087736],
    "ROC_AUC": [0.9619836996649984, 0.929947567535715, 0.8591875642680877, 0.9576031771253364, 0.9521417222697106, 0.8519301206738243, 0.9651488465794126],
    "Mean_CV_Score": [0.9094774196071616, 0.8664406291455851, 0.9009012545829951, 0.8935119313025908, 0.8845120009825062, 0.7803575718024598, 0.9044058478529342],
    "Precision_class_0": [0.92, 0.96, 0.9, 0.9, 0.87, 0.9, 0.91],
    "Recall_class_0": [0.76, 0.63, 0.76, 0.77, 0.81, 0.45, 0.79],
    "F1_score_class_0": [0.83, 0.76, 0.82, 0.83, 0.84, 0.6, 0.84],
    "Precision_class_1": [0.83, 0.76, 0.82, 0.83, 0.86, 0.68, 0.84],
    "Recall_class_1": [0.95, 0.98, 0.93, 0.93, 0.9, 0.96, 0.94],
    "F1_score_class_1": [0.88, 0.86, 0.87, 0.88, 0.88, 0.8, 0.89],
    "fnr": [0.054049284122747024, 0.018950340841403956, 0.06821240265613625, 0.06883010876039622, 0.09682543184274968, 0.041121577797877736, 0.0618147322905866]
}

freq_Percentage_Correlation_gini_index_accuracies = pd.DataFrame(data)

# New data for KNeighborsClassifier
new_data = {
    "Model": ["KNeighborsClassifier"],
    "Accuracy": [0.8387406318704678],
    "ROC_AUC": [0.9130038249487454],
    "Mean_CV_Score": [0.8904839434620271],
    "Precision_class_0": [0.87],
    "Recall_class_0": [0.76],
    "F1_score_class_0": [0.81],
    "Precision_class_1": [0.82],
    "Recall_class_1": [0.9],
    "F1_score_class_1": [0.86],
    "fnr": [0.09578856802488474]
}

# Convert new data to DataFrame
new_df = pd.DataFrame(new_data)

# Concatenate existing and new DataFrames
freq_Percentage_Correlation_gini_index_accuracies = pd.concat([freq_Percentage_Correlation_gini_index_accuracies, new_df], ignore_index=True)

# Display the combined DataFrame
print(freq_Percentage_Correlation_gini_index_accuracies)

#%%

# Extract accuracy columns from each DataFrame
accuracy_random_logically = freq_Random_logically_Gini_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_random_logically'})
accuracy_random_corr = freq_Random_Correlation_Gini_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_random_corr'})
accuracy_percentage_logically = freq_Percentage_Logically_Gini_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_percentage_logically'})
accuracy_percentage_corr = freq_Percentage_Correlation_gini_index_accuracies[['Model', 'Accuracy']].rename(columns={'Accuracy': 'Accuracy_percentage_corr'})

# Concatenate accuracy columns into one DataFrame
combined_accuracy_df = pd.concat([accuracy_random_logically.set_index('Model'), 
                                  accuracy_percentage_logically.set_index('Model'), 
                                  accuracy_random_corr.set_index('Model'), 
                                  accuracy_percentage_corr.set_index('Model')], axis=1)

# Display the combined accuracy DataFrame
print(combined_accuracy_df)



#%%
import pandas as pd

data = {
    'Model': ['eXtreme Gradient Boosting (XGBoost)', 'AdaBoostClassifier', 'RandomForestClassifier', 'MLPClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'GaussianNB'],
    'Accuracy_random_logically': [0.8535806247874459, 0.8691881649905262, 0.796725453043774, 0.5494339989311567, 0.8387990088908323, 0.5515838313171063, 0.4521328280619929, 0.5541587718019725],
    'Accuracy_percentage_logically': [0.8687186463736077, 0.8521870103368275, 0.8639085597677554, 0.8677104716557144, 0.8531344516379803, 0.8387892186038602, 0.8335661447641721, 0.733380300508946],
    'Accuracy_random_corr': [0.7917091774765583, 0.7825875722683768, 0.8676942136714765, 0.7473886216780838, 0.8478355924792305, 0.7368580867706359, 0.7136957683525239, 0.7174974493514065],
    'Accuracy_percentage_corr': [0.8702248351087736, 0.8634105457504828, 0.861685716715051, 0.8597665407460493, 0.8527093177207964, 0.8387406318704678, 0.821820301966548, 0.7289953478202776]
}

df = pd.DataFrame(data).sort_values('Model')
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
df['Model'] = df['Model'].map(model_abbreviations)
print(df)


#%%
# Maximum values in each case
# Maximum values in each case
max_values = df.max()

# Minimum values in each case
min_values = df.min()

print("Maximum Values:")
for column_name, max_value in max_values.iteritems():
    print(f"{column_name}: {max_value} (Model: {df.loc[df[column_name] == max_value, 'Model'].values[0]})")

print("\nMinimum Values:")
for column_name, min_value in min_values.iteritems():
    print(f"{column_name}: {min_value} (Model: {df.loc[df[column_name] == min_value, 'Model'].values[0]})")

#%%
import pandas as pd

data = {
    'Model': ['eXtreme Gradient Boosting (XGBoost)', 'AdaBoostClassifier', 'RandomForestClassifier', 'MLPClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'GaussianNB'],
    'Randomly Relation based': [0.8535806247874459, 0.8691881649905262, 0.796725453043774, 0.8117221567650953, 0.8387990088908323, 0.5515838313171063, 0.790636972074775, 0.5541587718019725],
    'Most Frequent Relation': [0.8687186463736077, 0.8521870103368275, 0.8639085597677554, 0.8677104716557144, 0.8531344516379803, 0.8387892186038602, 0.8335661447641721, 0.733380300508946],
    'Random Correlated': [0.7917091774765583, 0.7825875722683768, 0.8676942136714765, 0.7473886216780838, 0.8478355924792305, 0.7368580867706359, 0.7136957683525239, 0.7174974493514065],
    'Most Frequent Correlated': [0.8702248351087736, 0.8634105457504828, 0.861685716715051, 0.8597665407460493, 0.8527093177207964, 0.8387406318704678, 0.821820301966548, 0.7289953478202776]
}

df = pd.DataFrame(data).sort_values('Model')
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
df['Model'] = df['Model'].map(model_abbreviations)
print(df)


import matplotlib.pyplot as plt

# Set the 'Model' column as the index for easier plotting
df.set_index('Model', inplace=True)

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
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 
        plt.tight_layout()
    plt.tight_layout()
    # Set title over the overall plot
    fig.suptitle('Gini Index', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plots
    plt.show()

# Create sets of 4 graphs
for i in range(0, len(df.columns), 4):
    create_graphs(df.iloc[:, i:i+4])


#%%

# Extract 'fnr' columns from each DataFrame
import pandas as pd

# Abbreviation mapping for model names
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
freq_Random_logically_Gini_accuracies['Model'] = freq_Random_logically_Gini_accuracies['Model'].map(model_abbreviations)
freq_Random_Correlation_Gini_accuracies['Model'] = freq_Random_Correlation_Gini_accuracies['Model'].map(model_abbreviations)
freq_Percentage_Logically_Gini_accuracies['Model'] = freq_Percentage_Logically_Gini_accuracies['Model'].map(model_abbreviations)
freq_Percentage_Correlation_gini_index_accuracies['Model'] = freq_Percentage_Correlation_gini_index_accuracies['Model'].map(model_abbreviations)

# Extract 'fnr' columns from each DataFrame and rename them
fnr_random_logically = freq_Random_logically_Gini_accuracies.rename(columns={'fnr': 'FNR_random_logically'})
fnr_random_corr = freq_Random_Correlation_Gini_accuracies.rename(columns={'fnr': 'FNR_random_corr'})
fnr_percentage_logically = freq_Percentage_Logically_Gini_accuracies.rename(columns={'fnr': 'FNR_percentage_logically'})
fnr_percentage_corr = freq_Percentage_Correlation_gini_index_accuracies.rename(columns={'fnr': 'FNR_percentage_corr'})

# Concatenate 'fnr' columns into one DataFrame
combined_fnr_df = pd.concat([fnr_random_logically.set_index('Model'), 
                             fnr_percentage_logically.set_index('Model'), 
                             fnr_random_corr.set_index('Model'), 
                             fnr_percentage_corr.set_index('Model')], axis=1)

# Display the combined 'fnr' DataFrame
print(combined_fnr_df)

#%%
import matplotlib.pyplot as plt

# Set the 'Model' column as the index for easier plotting
df.set_index('fnr', inplace=True)

# Define a function to create a set of 4 graphs
import matplotlib.pyplot as plt

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
        ax.set_ylabel('FNR')  # Change ylabel to 'FNR'
        ax.set_xlabel('Model')  # Change xlabel to 'Model'
        ax.set_ylim(0, 1)  # Set y-axis limits
        ax.grid(True)
        plt.xticks(rotation=45, ha='right')
        ax.set_xticklabels(dataframe.index, rotation=45)  # Set x-axis labels to model names
        
    # Set title over the overall plot
    fig.suptitle('FNR Gini', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plots
    plt.show()

# Create sets of 4 graphs
for i in range(0, len(df.columns), 4):
    create_graphs(df.iloc[:, i:i+4])

