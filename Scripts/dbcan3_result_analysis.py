#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-27
# Description:
#   Analyzes family-specific CSV files to compute prediction metrics based on
#   #ofTools threshold (≥2 tools). Calculates precision, recall, accuracy, and
#   F1 score assuming all entries are true positives.
#
# Usage:
#   1. Configure directory_path and excel_output_path in script
#   2. python dbcan3_result_analysis.py
#
# Input:
#   CSV files starting with 'AA' and containing '#ofTools' column
# Output:
#   Excel file with metrics summary per family (TP, FP, TN, FN, precision, recall, accuracy, F1)

import pandas as pd
import os

directory_path = "path/to/your/directory"

files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.startswith("AA") and f.endswith(".csv")]

# Function to analyze each file
def analyze_file(file_path):
    df = pd.read_csv(file_path)
    
    # Prediction rule: Positive if #ofTools >= 2
    df['Predicted'] = df['#ofTools'] >= 2
    
    # All entries are actual positives
    df['Actual'] = True

    # Compute counts
    TP = ((df['Predicted'] == True) & (df['Actual'] == True)).sum()
    FN = ((df['Predicted'] == False) & (df['Actual'] == True)).sum()

    # Assume FP and TN are 0 since no negatives are present
    FP = 0
    TN = 0

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Family": os.path.basename(file_path).replace(".csv", ""),
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "PRECISION": round(precision, 4),
        "RECALL": round(recall, 4),
        "ACCURACY": round(accuracy, 4),
        "F1 SCORE": round(f1, 4)
    }

# Analyze all files
results = [analyze_file(file) for file in files]

# Create DataFrame and save to Excel
results_df = pd.DataFrame(results)
excel_output_path = "path/to/your/output_directory"
results_df.to_excel(excel_output_path, index=False)

excel_output_path
