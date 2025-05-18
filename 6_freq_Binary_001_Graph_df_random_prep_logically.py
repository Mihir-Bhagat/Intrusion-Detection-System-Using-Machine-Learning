
# #%%

#model for Binary output
# #%%
# import pandas as pd
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression

# def train_and_evaluate_model(train_data, test_data, columns_to_include_binary, model):
#     # Apply frequency encoding to categorical columns in train_data
#     # Separate features and target variable
#     X_train = train_data[columns_to_include_binary]
#     y_train = train_data['label']
#     X_test = test_data[columns_to_include_binary]
#     y_test = test_data['label']

#     # Scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     for column in X_train_scaled.columns:
#         if X_train_scaled[column].dtype == 'object':
#             encoding = train_data.groupby(column).size() / len(train_data)
#             X_train_scaled[column] = X_train_scaled[column].map(encoding)

#     # Apply frequency encoding to categorical columns in test_data
#     for column in X_test_scaled.columns:
#         if X_test_scaled[column].dtype == 'object':
#             encoding = train_data.groupby(column).size() / len(train_data)
#             X_test_scaled[column] = X_test_scaled[column].map(encoding)
            
#     for column in y_train.columns:
#         if y_train[column].dtype == 'object':
#             encoding = train_data.groupby(column).size() / len(train_data)
#             y_train[column] = y_train[column].map(encoding)

#     # Apply frequency encoding to categorical columns in test_data
#     for column in y_test.columns:
#         if y_test[column].dtype == 'object':
#             encoding = train_data.groupby(column).size() / len(train_data)
#             y_test[column] = y_test[column].map(encoding)    

#     # Train the model
#     clf = model
#     clf.fit(X_train_scaled, y_train)
#     predictions = clf.predict(X_test_scaled)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, predictions)

#     # Calculate confusion matrix
#     cm = confusion_matrix(y_test, predictions)

#     # Calculate False Negative Rate (FNR)
#     fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])

#     # Calculate classification report
#     classification_rep = classification_report(y_test, predictions)

#     # Extracting values using regular expressions
#     values = re.findall(r'\d+\.\d+', classification_rep)
#     precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1 = map(float, values[:6])

#     # Calculate ROC AUC score
#     roc_auc = roc_auc_score(y_test, predictions)

#     # Perform 5-fold cross-validation
#     cv_scores = cross_val_score(clf, np.concatenate((X_train_scaled, X_test_scaled)), np.concatenate((y_train, y_test)), cv=5, scoring='accuracy')
    
#     # Calculate the mean of cross-validation scores
#     mean_cv_score = np.mean(cv_scores)

#     # Predict probabilities instead of classes
#     probabilities = clf.predict_proba(X_test_scaled)[:, 1]

#     # Calculate ROC curve
#     fpr, tpr, thresholds = roc_curve(y_test, probabilities)

#     return accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr

# # Example usage:
# # accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data, test_data, columns_to_include_binary, LogisticRegression())

#%%
#%%


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def train_and_evaluate_model(train_data, test_data, columns_to_include_binary, model):
    # Separate features and target variable
    X_train = train_data[columns_to_include_binary].copy()  # Create a copy
    y_train = train_data['label']
    X_test = test_data[columns_to_include_binary].copy()    # Create a copy
    y_test = test_data['label']

    # Select only numeric columns for scaling
    # numeric_columns = X_train.select_dtypes(include=np.number).columns
    X_train_numeric = X_train #X_train[numeric_columns].copy()  # Create a copy
    X_test_numeric = X_test    # Create a copy

    # # Scale the numeric features
    # scaler = StandardScaler()
    X_train_scaled_numeric = X_train_numeric #scaler.fit_transform(X_train_numeric)
    X_test_scaled_numeric=X_test_numeric #= scaler.transform(X_test_numeric)

    # Replace scaled numeric columns in original DataFrames
    # X_train.loc[:, numeric_columns] = X_train_scaled_numeric
    # X_test.loc[:, numeric_columns] = X_test_scaled_numeric
    # # Apply frequency encoding to categorical columns in train_data
    # for column in X_train.columns:
    #     if X_train[column].dtype == 'object':
    #         encoding = train_data.groupby(column).size() / len(train_data)
    #         X_train[column] = X_train[column].map(encoding)

    
    # # Apply frequency encoding to categorical columns in test_data using encoding values from train_data
    # for column in X_test.columns:
    #     if X_test[column].dtype == 'object':
    #         encoding = train_data.groupby(column).size() / len(train_data)
    #         X_test[column] = X_test[column].map(encoding)
            

    nan_indices = X_test[X_test['state'].isnull()].index
    
    X_test.dropna(inplace=True)        

    # Drop corresponding rows from X_train
    X_train.drop(index=nan_indices, inplace=True)
    # Drop corresponding rows from y_train
    y_train.drop(index=nan_indices, inplace=True)
    # Drop corresponding rows from y_test
    y_test.drop(index=nan_indices, inplace=True)


    # Train the model with updated categorical columns
    clf = model
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Calculate False Negative Rate (FNR)
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    # Calculate classification report
    classification_rep = classification_report(y_test, predictions)

    # Extracting values using regular expressions
    values = re.findall(r'\d+\.\d+', classification_rep)
    precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1 = map(float, values[:6])

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, predictions)

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(clf, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), cv=5, scoring='accuracy')
    
    # Calculate the mean of cross-validation scores
    mean_cv_score = np.mean(cv_scores)

    # Predict probabilities instead of classes
    probabilities = clf.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)


    return accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr

#%%
#ACCURACIES

Accuracy_info_gain_new={}
Accuracy_gini_value_new={}
# from sklearn import metrics

#%%
# def model_prep(train_data, test_data, columns_to_include_binary, model):


#%%

#Preprocessed_data_name: df_random_prep_logically
#%%
                                                '''#RandomForestClassifier'''
from sklearn.ensemble import RandomForestClassifier
#%%
#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= RandomForestClassifier(n_estimators=100, random_state=42))

# Accuracy_info_gain_new['RandomForestClassifier']=accuracy, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr
accuracy_df_info_gain_threshold=accuracy

print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for RandomForestClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: RandomForestClassifier _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['RandomForestClassifier']=accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr

#%%

#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically, test_data=df_testing_random_prep_logically, columns_to_include_binary=thresholded_dict1_k_gini_value.keys(), model= RandomForestClassifier(n_estimators=100, random_state=42) )
accuracy_df_gini_thereshold=accuracy

Accuracy_gini_value_new['RandomForestClassifier']=accuracy, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for RandomForestClassifier:', roc_auc)


# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: RandomForestClassifier _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")


Accuracy_gini_value_new['RandomForestClassifier']=accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr


#%%
#Preprocessed_data_name: df_random_prep_logically
#%%
                                                    '''LogisticRegression'''
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= LogisticRegression())
# Accuracy_gini_value_new['LogisticRegression']=accuracy
accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for LogisticRegression:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: LogisticRegression _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['LogisticRegression']=accuracy_df_info_gain_threshold,roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr

#%%

#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fnr, probabilities,fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= LogisticRegression())
# Accuracy_gini_value_new['LogisticRegression']=accuracy
accuracy_df_gini_thereshold=accuracy

print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for LogisticRegression:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: LogisticRegression _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_gini_value_new['LogisticRegression']=accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%

#Preprocessed_data_name: df_random_prep_logically
#%%
                                            '''Decision Tree Classifier'''

from sklearn.tree import DecisionTreeClassifier
#%%
#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= DecisionTreeClassifier())

accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for DecisionTreeClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: DecisionTreeClassifier _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['DecisionTreeClassifier'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%
#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= DecisionTreeClassifier())
accuracy_df_gini_thereshold=accuracy
print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for DecisionTreeClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: DecisionTreeClassifier _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# For DecisionTreeClassifier

Accuracy_gini_value_new['DecisionTreeClassifier'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr


#%%
#Preprocessed_data_name: df_random_prep_logically
                                        '''K-Nearest-Neighbor (KNN) Classifier'''
#%%
from sklearn.neighbors import KNeighborsClassifier


#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= KNeighborsClassifier())

accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for KNeighborsClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: KNeighborsClassifier _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['KNeighborsClassifier'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%
#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= KNeighborsClassifier())
accuracy_df_gini_thereshold=accuracy

print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for KNeighborsClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: KNeighborsClassifier _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")


# For KNeighborsClassifier

Accuracy_gini_value_new['KNeighborsClassifier'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr



#%%

#Preprocessed_data_name: df_random_prep_logically
#%%
                                                '''Linear Support Vector Machine (SVM)'''

# from sklearn.svm import LinearSVC
# #%%
# #Information Gain: 

# #for threshold value >0.1
# accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= LinearSVC())

# accuracy_df_info_gain_threshold=accuracy
# print("Accuracy for threshold value >0.1:", accuracy)
# print("Classification Report:")
# print(classification_rep)
# print('ROC AUC for LinearSVC:', roc_auc)

# # Create the ROC curve
# plt.plot(fpr, tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('ROC curve: LinearSVC _Info_Gain ')
# plt.suptitle('df_random_prep_logically',fontsize=9)
# plt.show()

# roc_auc = metrics.auc(fpr, tpr)
# print(f"AUC: {roc_auc:.2f}")

# Accuracy_info_gain_new['LinearSVC'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
# #%%
# #Gini Value

# #for threshold value <=0.01
# accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
#                                                         test_data=df_testing_random_prep_logically.copy(),
#                                                         columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= LinearSVC())
# accuracy_df_gini_thereshold=accuracy
# print("Accuracy for threshold value <=0.01:", accuracy)
# print("Classification Report:")
# print(classification_rep)
# print('ROC AUC for LinearSVC:', roc_auc)

# # Create the ROC curve
# plt.plot(fpr, tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('ROC curve: LinearSVC _Gini_Index ')
# plt.suptitle('df_random_prep_logically',fontsize=9)
# plt.show()

# roc_auc = metrics.auc(fpr, tpr)
# print(f"AUC: {roc_auc:.2f}")

# # For LinearSVC

# Accuracy_gini_value_new['LinearSVC'] =accuracy_df_gini_thereshold,roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr


#%%

#Preprocessed_data_name: df_random_prep_logically
#%%%
                                                    '''# Multi-layer Perceptron (MLP)'''
from sklearn.neural_network import MLPClassifier
#%%%
#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= MLPClassifier())

accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for MLPClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: MLPClassifier _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['MLPClassifier'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%

#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= MLPClassifier())
accuracy_df_gini_thereshold=accuracy
print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for MLPClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: MLPClassifier _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# For MLPClassifier

Accuracy_gini_value_new['MLPClassifier'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr


#%%

#Preprocessed_data_name: df_random_prep_logically
#%%
                                                '''# AdaBoost Ensemble Learning Method'''
from sklearn.ensemble import AdaBoostClassifier
#%%%
#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= AdaBoostClassifier())

accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for AdaBoostClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: AdaBoostClassifier _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['AdaBoostClassifier'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%
#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= AdaBoostClassifier())
accuracy_df_gini_thereshold=accuracy
print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for AdaBoostClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: AdaBoostClassifier _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# For AdaBoostClassifier

Accuracy_gini_value_new['AdaBoostClassifier'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr

#%%

#Preprocessed_data_name: df_random_prep_logically
#%%%
                                                            '''# Naïve Bayes (NB)'''
from sklearn.naive_bayes import GaussianNB
#%%%

#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= GaussianNB())

accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for GaussianNB:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: GaussianNB _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

Accuracy_info_gain_new['GaussianNB'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%
#Gini Value

#for threshold value <=0.01
tpr_infoaccuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= GaussianNB())
accuracy_df_gini_thereshold=accuracy
print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for GaussianNB:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: GaussianNB _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# For GaussianNB

Accuracy_gini_value_new['GaussianNB'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr

#%%

#Preprocessed_data_name: df_random_prep_logically
#%%%
                                        '''# eXtreme Gradient Boosting (XGBoost)'''
import xgboost as xgb
#%%%

#Information Gain: 

#for threshold value >0.1
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= xgb.XGBClassifier())

accuracy_df_info_gain_threshold=accuracy
print("Accuracy for threshold value >0.1:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for xgb.XGBClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: xgb.XGBClassifier _Info_Gain ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")
Accuracy_info_gain_new['eXtreme Gradient Boosting (XGBoost)'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
#%%
#Gini Value

#for threshold value <=0.01
accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
                                                        test_data=df_testing_random_prep_logically.copy(),
                                                        columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= xgb.XGBClassifier())
accuracy_df_gini_thereshold=accuracy
print("Accuracy for threshold value <=0.01:", accuracy)
print("Classification Report:")
print(classification_rep)
print('ROC AUC for xgb.XGBClassifier:', roc_auc)

# Create the ROC curve
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve: xgb.XGBClassifier _Gini_Index ')
plt.suptitle('df_random_prep_logically',fontsize=9)
plt.show()

roc_auc = metrics.auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

# For eXtreme Gradient Boosting (XGBoost)

Accuracy_gini_value_new['eXtreme Gradient Boosting (XGBoost)'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr


#%%

#Preprocessed_data_name: df_random_prep_logically
#%%
                                                '''# Gaussian Mixture Models (GMM)'''
# from sklearn.mixture import GaussianMixture
# #%%%
# #Information Gain: 

# #for threshold value >0.1
# accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(), test_data=df_testing_random_prep_logically.copy(), columns_to_include_binary=thresholded_dict1_k_info_gain_value.keys(), model= GaussianMixture())

# accuracy_df_info_gain_threshold=accuracy
# print("Accuracy for threshold value >0.1:", accuracy)
# print("Classification Report:")
# print(classification_rep)
# print('ROC AUC for GaussianMixture:', roc_auc)

# # Create the ROC curve
# plt.plot(fpr, tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('ROC curve: GaussianMixture _Info_Gain ')
# plt.suptitle('df_random_prep_logically',fontsize=9)
# plt.show()

# roc_auc = metrics.auc(fpr, tpr)
# print(f"AUC: {roc_auc:.2f}")
# Accuracy_info_gain_new['GaussianMixture'] =accuracy_df_info_gain_threshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr
# #%%
# #Gini Value

# #for threshold value <=0.01
# accuracy, classification_rep, roc_auc, mean_cv_score, precision_class_0, recall_class_0, f1_score_class_0, precision_class_1, recall_class_1, f1_score_class_1, fnr, probabilities, fpr, tpr = train_and_evaluate_model(train_data=df_random_prep_logically.copy(),
#                                                         test_data=df_testing_random_prep_logically.copy(),
#                                                         columns_to_include_binary=thresholded_dict1_k_gini_value.keys(),model= GaussianMixture())
# accuracy_df_gini_thereshold=accuracy
# print("Accuracy for threshold value <=0.01:", accuracy)
# print("Classification Report:")
# print(classification_rep)
# print('ROC AUC for GaussianMixture:', roc_auc)

# # Create the ROC curve
# plt.plot(fpr, tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('ROC curve: GaussianMixture _Gini_Index ')
# plt.suptitle('df_percentage_prep_logically')
# plt.show()

# roc_auc = metrics.auc(fpr, tpr)
# print(f"AUC: {roc_auc:.2f}")

# # For GaussianMixture

# Accuracy_gini_value_new['GaussianMixture'] =accuracy_df_gini_thereshold, roc_auc, mean_cv_score, precision_class_0,recall_class_0,f1_score_class_0,precision_class_1,recall_class_1,f1_score_class_1,fpr, tpr, fnr


#%%
import pandas as pd

# Assuming Accuracy_info_gain_new is a dictionary containing arrays of different lengths
# Check the lengths of all arrays
lengths = {key: len(value) for key, value in Accuracy_info_gain_new.items()}
print("Lengths of arrays:", lengths)

# Find the minimum length among the arrays
min_length = min(lengths.values())

# Truncate longer arrays or pad shorter arrays to match the minimum length
adjusted_data = {key: value[:min_length] for key, value in Accuracy_info_gain_new.items()}

# Create the DataFrame
A_info_gain = pd.DataFrame(adjusted_data).T






A_info_gain=pd.DataFrame(Accuracy_info_gain_new).T
A_info_gain.columns=['Accuracy', 'ROC_AUC', 'Mean_CV_Score', 'Precision_class_0', 'Recall_class_0', 'F1_score_class_0', 'Precision_class_1', 'Recall_class_1', 'F1_score_class_1','fpr', 'tpr','fnr']
A_info_gain_rand_logic = A_info_gain.apply(pd.to_numeric, errors='coerce')
# A_info_gain.info()

#%%

# import pandas as pd

# # Assuming Accuracy_info_gain_new is a dictionary containing arrays of different lengths
# # Check the lengths of all arrays
# lengths = {key: len(value) for key, value in Accuracy_gini_value_new.items()}
# print("Lengths of arrays:", lengths)

# # Find the minimum length among the arrays
# min_length = min(lengths.values())

# # Truncate longer arrays or pad shorter arrays to match the minimum length
# adjusted_data = {key: value[:min_length] for key, value in Accuracy_gini_value_new.items()}

# # Create the DataFrame
# A_gini_index = pd.DataFrame(adjusted_data).T

A_gini_index=pd.DataFrame(Accuracy_gini_value_new).T
A_gini_index.columns=['Accuracy', 'ROC_AUC', 'Mean_CV_Score', 'Precision_class_0', 'Recall_class_0', 'F1_score_class_0', 'Precision_class_1', 'Recall_class_1', 'F1_score_class_1','fpr', 'tpr','fnr']
A_gini_index_rand_logic = A_gini_index.apply(pd.to_numeric, errors='coerce')

