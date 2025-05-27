#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-22
# Description:
#   Implements an independent set prediction pipeline for multiclass classification with:
#   - Model/imputer/scaler loading from joblib files
#   - Automated data preprocessing and missing value handling
#   - Prediction generation with optional true label evaluation
#   - Comprehensive reporting (classification metrics, confusion matrix, ROC/PR curves)
#
# Usage:
#   Configure base_path in the script
#   python multiclass_model_evaluation_modified_huber.py
#   (Results saved to specified output directory)
#
# Input:
#   independent_testset.csv with features and optional 'class' label column
# Output:
#   predictions.csv (original data + predicted classes)
#   independent_report.txt (classification metrics)
#   independent_cm.png (confusion matrix)
#   independent_pr.png (precision-recall curves)
#   independent_roc.png (ROC curves)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc

# Configure paths
base_path = "path/to/your/independent_set"
model_dir = os.path.join(base_path, "modified_huber")
output_dir = os.path.join(model_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

def load_artifacts():
    model = joblib.load(os.path.join(model_dir, "multiclass_model.joblib"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, imputer, scaler

def predict_and_save(model, imputer, scaler):
    print("\n Loading independent set")
    df = pd.read_csv(os.path.join(base_path, "independent_testset.csv"))
    
    # Store true labels if available
    if 'class' in df.columns:
        y_true = df.iloc[:, 0].values.astype(int)
        X = df.iloc[:, 1:]
    else:
        y_true = None
        X = df

    print("\n Preprocessing")
    X = imputer.transform(X)
    X = scaler.transform(X)

    print("\n Making predictions")
    y_pred = model.predict(X)
    y_score = model.decision_function(X) if hasattr(model, 'decision_function') else None

    # Save predictions
    df['predicted_class'] = y_pred
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    if y_true is not None:
        generate_reports(y_true, y_pred, model.classes_, y_score)

def generate_reports(y_true, y_pred, classes, y_score=None):
    print("\n Generating reports")
    
    # Classification Report with zero_division=0
    report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
    print("\n Classification Report:")
    print(report)
    with open(os.path.join(output_dir, "independent_report.txt"), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Independent Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, "independent_cm.png"), bbox_inches='tight')
    plt.close()

    # ROC/PR Curves if scores available
    if y_score is not None:
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # Precision-Recall Curve
        plt.figure(figsize=(10,8))
        for i in range(len(classes)):
            precision, recall, _ = precision_recall_curve(y_true_bin[:,i], y_score[:,i])
            plt.plot(recall, precision, lw=2, label=f'Class {classes[i]}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Independent Set)')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.savefig(os.path.join(output_dir, "independent_pr.png"), bbox_inches='tight')
        plt.close()

        # ROC Curve
        plt.figure(figsize=(10,8))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_score[:,i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC={roc_auc:.2f})')
        plt.plot([0,1],[0,1],'k--',lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Independent Set)')
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.savefig(os.path.join(output_dir, "independent_roc.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    model, imputer, scaler = load_artifacts()
    predict_and_save(model, imputer, scaler)
    print("\n Prediction results saved in:", output_dir)
