#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-27
# Description:
#   Performs ensemble feature selection (EFS) on a dataset using multiple statistical
#   and machine learning methods, including Mann-Whitney U test, Pearson/Spearman
#   correlation, Logistic Regression, and Random Forest (entropy and gini).
#   Outputs a CSV file with normalized importance scores for each feature.
#
# Usage:
#   python ensemble_feature_selection.py
#   (Then enter the input filename, e.g., sample_testing.csv, when prompted)
#
# Input:
#   A CSV file with features and binary class labels (0/1) in the first column.
# Output:
#   EFS_values_<input_basename>.csv in an output_files subdirectory.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
import time
import os

# Define base path
base_path = "path/to/your/directory"

# Prompt the user for the input filename
input_filename = input("Enter the input filename (e.g., sample_testing.csv): ")

# Construct the full input filepath
input_filepath = os.path.join(base_path, input_filename)

# Construct the output directory
output_dir = os.path.join(base_path, "output_files")

# Check if the output directory exists, and create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Extract filename without extension for constructing output filenames
filename_no_ext = os.path.splitext(input_filename)[0]

# Construct the output filename for the selected features dataset
output_filename = f"EFS_{filename_no_ext}.csv"
output_filepath = os.path.join(output_dir, output_filename)

# Construct the output filename for the values dataset
values_output_filename = f"EFS_values_{filename_no_ext}.csv"
values_output_filepath = os.path.join(output_dir, values_output_filename)

# Start measuring time
start_time = time.time()

# Preprocess data function
def preprocess_data(data):
    # Convert class column to numeric if necessary
    if data.iloc[:, 0].dtype == 'object':
        data.iloc[:, 0] = pd.Categorical(data.iloc[:, 0]).codes
    
    # Apply jitter to numeric features but preserve zeros
    numeric_features = data.iloc[:, 1:].select_dtypes(include=['int64', 'float64']).columns
    data[numeric_features] = data[numeric_features].apply(lambda x: np.where(x != 0, x + np.random.normal(0, 1e-6, len(x)), x))
    
    # Remove NA rows if necessary
    data = data.dropna()
    
    return data

# Define the ensemble_fs function
def ensemble_fs(data, class_column_number, NA_threshold=0.2, cor_threshold=0.7, runs=100, selection=[True, True, True, True, True, True]):
    start_time = time.time()
    
    if class_column_number is None:
        raise ValueError("class_column_number argument missing")
    
    if not isinstance(NA_threshold, (int, float)) or NA_threshold > 1 or NA_threshold < 0:
        raise ValueError("Invalid argument: NA_threshold must be in [0,1]")
    
    if not isinstance(cor_threshold, (int, float)) or cor_threshold > 1 or cor_threshold < 0:
        raise ValueError("Invalid argument: cor_threshold must be in [0,1]")
    
    if not isinstance(runs, int) or runs <= 0:
        raise ValueError("Invalid argument: runs must be a positive integer")
    
    if not all(isinstance(x, bool) for x in selection):
        raise ValueError("Invalid argument: selection must be a list of boolean values")
    
    # Remove columns with too many NAs
    na_counts = data.isnull().sum() / len(data)
    data = data.loc[:, na_counts <= NA_threshold]
    
    # Remove rows with any NAs
    data = data.dropna()
    
    # Remove columns with all zeros or zero variance
    data = data.loc[:, (data != 0).any(axis=0) & (data.var() != 0)]
    
    # Round class labels to nearest integer
    data.iloc[:, class_column_number] = np.round(data.iloc[:, class_column_number])
    
    # Check for binary class
    class_labels = set(data.iloc[:, class_column_number])
    
    if len(class_labels) != 2 or (0 not in class_labels or 1 not in class_labels):
        print(f"Class labels are not binary with labels 0 and 1. Found labels: {class_labels}")
        raise ValueError("Class not binary with classlabels 0 and 1")
    
    # Feature selection methods
    X = data.iloc[:, [i for i in range(len(data.columns)) if i != class_column_number]]
    y = data.iloc[:, class_column_number]
    
    # Median filter
    if selection[0]:
        print('Start Median')
        positives = y[y == 1].index
        negatives = y[y == 0].index
        
        data_pos = X.loc[positives]
        data_neg = X.loc[negatives]
        
        mannwhitneyu_scores = []
        for col in X.columns:
            _, pval = mannwhitneyu(data_pos[col], data_neg[col])
            mannwhitneyu_scores.append(1 - pval)
        
        imp1 = np.array(mannwhitneyu_scores)
    else:
        imp1 = np.zeros(len(X.columns))
    
    # Pearson correlation
    if selection[1]:
        print('Start Pearson')
        pearson_scores = np.array([np.corrcoef(X[col], y)[0, 1] for col in X.columns])
        imp2 = np.abs(pearson_scores)
    else:
        imp2 = np.zeros(len(X.columns))
    
    # Spearman correlation
    if selection[2]:
        print('Start Spearman')
        spearman_scores = np.array([np.corrcoef(np.argsort(X[col]), np.argsort(y))[0, 1] for col in X.columns])
        imp3 = np.abs(spearman_scores)
    else:
        imp3 = np.zeros(len(X.columns))
    
    # Logistic Regression
    if selection[3]:
        print('Start LogReg')
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X, y)
        imp4 = np.abs(logreg.coef_[0])
    else:
        imp4 = np.zeros(len(X.columns))
    
    # RandomForest - Modified Section
    imp_rf_entropy = np.zeros(len(X.columns))
    imp_rf_gini = np.zeros(len(X.columns))
    
    if selection[4] or selection[5]:
        print('Start RF')
        if selection[4]:
            rf_entropy = RandomForestClassifier(n_estimators=5000, criterion='entropy', random_state=42)
            rf_entropy.fit(X, y)
            imp_rf_entropy = rf_entropy.feature_importances_
        
        if selection[5]:
            rf_gini = RandomForestClassifier(n_estimators=5000, criterion='gini', random_state=42)
            rf_gini.fit(X, y)
            imp_rf_gini = rf_gini.feature_importances_
    
    # Normalize scores to [0, 1/n]
    n = sum(selection)  
    epsilon = 1e-9  
    
    imp1_normalized = imp1 / (np.max(imp1) + epsilon) * (1/n) if selection[0] else np.zeros(len(X.columns))
    imp2_normalized = imp2 / (np.max(imp2) + epsilon) * (1/n) if selection[1] else np.zeros(len(X.columns))
    imp3_normalized = imp3 / (np.max(imp3) + epsilon) * (1/n) if selection[2] else np.zeros(len(X.columns))
    imp4_normalized = imp4 / (np.max(imp4) + epsilon) * (1/n) if selection[3] else np.zeros(len(X.columns))
    imp_rf_entropy_normalized = imp_rf_entropy / (np.max(imp_rf_entropy) + epsilon) * (1/n) if selection[4] else np.zeros(len(X.columns))
    imp_rf_gini_normalized = imp_rf_gini / (np.max(imp_rf_gini) + epsilon) * (1/n) if selection[5] else np.zeros(len(X.columns))
    
    # Calculate EFS score
    efs_scores = (imp1_normalized + imp2_normalized + imp3_normalized +
                 imp4_normalized + imp_rf_entropy_normalized + imp_rf_gini_normalized)
    
    # Create output dataframe
    feature_names = X.columns
    output_data = pd.DataFrame({
        'Feature': feature_names,
        'Median': imp1_normalized,
        'Pearson Correlation': imp2_normalized,
        'Spearman Correlation': imp3_normalized,
        'Logistic Regression': imp4_normalized,
        'Random Forest (Entropy)': imp_rf_entropy_normalized,
        'Random Forest (Gini)': imp_rf_gini_normalized,
        'EFS Score': efs_scores,
    })
    
    return output_data

def main():
    data = pd.read_csv(input_filepath)
    data = preprocess_data(data)    
    print("Class labels:", set(data.iloc[:, 0]))
    data.iloc[:, 0] = np.round(data.iloc[:, 0])    
    print("Class labels after rounding:", set(data.iloc[:, 0]))    
    output_data = ensemble_fs(data, class_column_number=0, runs=100)  
    output_data.to_csv(values_output_filepath, index=False)    
    print(f"Results saved to: {values_output_filepath}")
if __name__ == "__main__":
    main()

