#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-04-25
# Description:
#   Implements a neural network for multiclass classification with:
#   - Automated data preprocessing and validation
#   - Class imbalance handling through weighted learning
#   - Advanced model architecture with dropout/batch normalization
#   - Comprehensive performance visualization (ROC, PR curves, confusion matrix)
#   - Model persistence and training history tracking
#
# Usage:
#   Configure BASE_PATH in script configuration section
#   python multiclass_neuralnet.py
#   (Training results saved to specified output directory)
#
# Input:
#   training_multiclass.csv (First column: labels, subsequent: features)
#   validation_multiclass.csv (Same format as training data)
# Output:
#   enhanced_multiclass_model.keras (Trained model)
#   scaler.joblib (Feature normalization scaler)
#   *.png (Performance visualizations)
#   classification_report.txt (Detailed metrics)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# Configuration
BASE_PATH = "path/to/your/directory"
INPUT_TRAIN = os.path.join(BASE_PATH, "training_multiclass.csv")
INPUT_TEST = os.path.join(BASE_PATH, "validation_multiclass.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "Multiclass_neuralnet_output")
CLASS_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess training/validation data"""
    train_df = pd.read_csv(INPUT_TRAIN)
    test_df = pd.read_csv(INPUT_TEST)

    # Data validation
    assert not train_df.isnull().any().any(), "Missing values in training data"
    assert not test_df.isnull().any().any(), "Missing values in test data"

    # Feature/label separation
    X_train = train_df.drop(columns=train_df.columns[0]).values
    y_train = train_df.iloc[:, 0].values.astype(int)
    X_test = test_df.drop(columns=test_df.columns[0]).values
    y_test = test_df.iloc[:, 0].values.astype(int)

    # Feature normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))

    return X_train, X_test, y_train, y_test

def create_balanced_model(input_shape):
    """Build neural network architecture"""
    model = tf.keras.Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(9, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_class_weights(y_train):
    """Calculate class weights for imbalanced data"""
    return dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    ))

def get_callbacks():
    """Configure training callbacks"""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            min_delta=0.001,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-6
        )
    ]

def generate_performance_report(model, X_test, y_test):
    """Generate evaluation metrics and visualizations"""
    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_score = model.predict(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(CLASS_LABELS)))

    # Confusion Matrix
    plt.figure(figsize=(12,10))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, digits=4)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("Multiclass Classification Report:\n\n")
        f.write(report)

    # Precision-Recall Curve
    plt.figure(figsize=(10,8))
    for i in range(len(CLASS_LABELS)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {CLASS_LABELS[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'), bbox_inches='tight')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(10,8))
    for i in range(len(CLASS_LABELS)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {CLASS_LABELS[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Calculate class weights
    class_weights = get_class_weights(y_train)
    print("Class Weights:", class_weights)

    # Create and train model
    model = create_balanced_model((X_train.shape[1],))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=128,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save model
    model.save(os.path.join(OUTPUT_DIR, 'enhanced_multiclass_model.keras'))

    # Generate reports and visualizations
    generate_performance_report(model, X_test, y_test)

    # Training history visualization
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()

    print(f"\nTraining complete! Results saved to: {OUTPUT_DIR}")
