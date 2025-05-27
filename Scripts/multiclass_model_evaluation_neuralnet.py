#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-13
# Description:
#   Performs multiclass predictions on independent datasets with:
#   - Pre-trained neural network model loading
#   - Automated feature scaling using saved scaler
#   - Prediction file generation with class labels
#   - Comprehensive reporting (classification metrics, confusion matrix)
#
# Usage:
#   1. Configure BASE_PATH and INDEPENDENT_DATA_PATH in script
#   2. python multiclass_model_evaluation_neuralnet.py
#   (Results saved to Predictions subdirectory)
#
# Input:
#   CSV file with features
# Output:
#   predictions.csv 
#   classification_report.txt 
#   confusion_matrix.png

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Configuration ====
BASE_PATH = "path/to/your/directory"
MODEL_PATH = os.path.join(BASE_PATH, "enhanced_multiclass_model.keras")
SCALER_PATH = os.path.join(BASE_PATH, "scaler.joblib")
PREDICTION_OUTPUT_DIR = os.path.join(BASE_PATH, "Predictions")
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# Path to your independent data CSV
INDEPENDENT_DATA_PATH = "path/to/your/independentset"

# Class labels (must match your model)
CLASS_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

# ==== Prediction and Reporting ====
def predict_and_report():
    # Load scaler and model
    scaler = load(SCALER_PATH)
    model = load_model(MODEL_PATH)

    # Load new data
    new_df = pd.read_csv(INDEPENDENT_DATA_PATH)
    # If the first column is labels, separate them
    if new_df.columns[0] in ['label', 'class', 'Class', '0', 0]:
        X_new = new_df.drop(columns=new_df.columns[0]).values
        y_true = new_df.iloc[:, 0].values.astype(int)
        has_labels = True
    else:
        X_new = new_df.values
        y_true = None
        has_labels = False

    # Scale features
    X_new_scaled = scaler.transform(X_new)

    # Predict
    y_pred_probs = model.predict(X_new_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Save predictions
    pred_df = pd.DataFrame({'Predicted_Class': y_pred})
    pred_df.to_csv(os.path.join(PREDICTION_OUTPUT_DIR, "predictions.csv"), index=False)

    # If labels are present, generate report and confusion matrix
    if has_labels:
        report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=4)
        with open(os.path.join(PREDICTION_OUTPUT_DIR, "classification_report.txt"), "w") as f:
            f.write("Classification Report on Independent Data\n\n")
            f.write(report)
        # Confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        plt.title('Confusion Matrix (Independent Data)')
        plt.savefig(os.path.join(PREDICTION_OUTPUT_DIR, "confusion_matrix.png"))
        plt.close()
        print("Classification report and confusion matrix saved.")
    else:
        print("Predictions saved. No true labels found, so no report generated.")

if __name__ == "__main__":
    predict_and_report()
    print(f"Done! Results are in: {PREDICTION_OUTPUT_DIR}")
