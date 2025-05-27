#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-13
# Description:
#   Independent Data Prediction Pipeline for BiLSTM Protein Classification:
#   - Loads pre-trained model and tokenizer
#   - Processes protein sequences with dynamic padding/truncation
#   - Generates multiclass predictions (0-8)
#   - Produces evaluation metrics and visualizations
#   - Saves predictions with original data
#
# Usage:
#   Configure model_path, tokenizer_path, independent_data_path, seq_length
#   python multiclass_model_evaluation_bilstm.py
#
# Input:
#   CSV file containing:
#   - 'Sequence' column: Protein sequence strings
#   - 'Class' column: True labels (0-8)
# Output:
#   independent_predictions.csv (Original data + predictions)
#   classification_report_independent.txt
#   confusion_matrix_independent.png

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_independent_data(
    model_path,
    tokenizer_path,
    independent_data_path,
    seq_length,
    output_dir
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load model
    model = load_model(model_path)

    # Load independent data
    independent_df = pd.read_csv(independent_data_path)
    assert 'Sequence' in independent_df.columns and 'Class' in independent_df.columns, \
        "CSV must have 'Sequence' and 'Class' columns"

    # Preprocess sequences
    X_independent = pad_sequences(
        tokenizer.texts_to_sequences(independent_df['Sequence']),
        maxlen=seq_length,
        padding='post',
        truncating='post'
    )

    # True labels
    y_true = independent_df['Class'].values

    # Predict probabilities and classes
    y_probs = model.predict(X_independent)
    y_pred = np.argmax(y_probs, axis=1)

    # Define all 9 class names (including class 7)
    class_names = [f"Class_{i}" for i in range(9)]
    all_labels = list(range(9))

    # Classification report (includes all classes)
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0  
    )
    print("Classification Report:\n", report)

    # Save the classification report as text
    with open(os.path.join(output_dir, "classification_report_independent.txt"), "w") as f:
        f.write(report)

    # Confusion matrix (includes all classes)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix for Independent Data')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_independent.png'))
    plt.show()
    independent_df['Predicted_Class'] = y_pred
    independent_df.to_csv(os.path.join(output_dir, 'independent_predictions.csv'), index=False)
if __name__ == "__main__":
    predict_independent_data(
        model_path="path/to/your/model_directory",
        tokenizer_path="path/to/your/tokenizer_directory",
        independent_data_path="path/to/your/independentset",
        seq_length=400,
        output_dir="path/to/your/output_directory""
    )
