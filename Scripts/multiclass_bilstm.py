#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-27
# Description:
#   Implements a BiLSTM model for multiclass protein sequence classification with:
#   - Character-level tokenization and dynamic sequence padding
#   - Class imbalance handling via balanced weighting
#   - Bidirectional LSTM architecture with layer normalization
#   - Comprehensive evaluation (ROC, PR curves, confusion matrix)
#   - Model checkpointing and training history tracking
#
# Usage:
#   Configure INPUT_DIR, OUTPUT_DIR, and SEQ_LENGTH in the script
#   python multiclass_bilstm.py
#   (Training results saved to specified output directory)
#
# Input:
#   training.csv and test.csv containing:
#   - 'Class' column: Integer labels (0-8)
#   - 'Sequence' column: Protein sequence strings
# Output:
#   best_model.keras (Trained BiLSTM model)
#   tokenizer.pkl (Serialized tokenizer)
#   *.png (Evaluation visualizations)
#   classification_report.txt (Performance metrics)

# Install required packages (run once)
# !pip install tensorflow scikit-learn pandas matplotlib seaborn numpy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, Bidirectional, 
    Dropout, Input, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2

def configure_model(vocab_size, seq_length, num_classes):
    """Optimized BiLSTM model architecture"""
    model = Sequential([
        Input(shape=(seq_length,)),
        Embedding(vocab_size, 256, mask_zero=True),
        Bidirectional(LSTM(512, return_sequences=True)),
        LayerNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(256)),
        LayerNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def save_model_and_reports(model, history, y_test, y_pred, y_probs, output_dir):
    """Generate comprehensive evaluation reports and visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Classification Report
    class_names = [f"Class_{i}" for i in range(9)]
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Confusion Matrix
    plt.figure(figsize=(14, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curves for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc = roc_auc_score(y_test == i, y_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
    # Precision-Recall Curves for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_test == i, y_probs[:, i])
        ap_score = average_precision_score(y_test == i, y_probs[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap_score:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'))
    plt.close()

    # Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Evolution')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Evolution')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Save metrics
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nOverall ROC AUC (OvR): {roc_auc_score(y_test, y_probs, multi_class='ovr'):.4f}")
        f.write(f"\nAverage Precision: {average_precision_score(y_test, y_probs):.4f}")

def train_model(training_path, validation_path, output_dir, seq_length=433):
    """Complete training pipeline"""
    # Load and validate data
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(validation_path)
    
    assert {'Class', 'Sequence'}.issubset(train_df.columns), "Missing required columns"
    
    # Tokenization and sequence processing
    tokenizer = Tokenizer(char_level=True, lower=False, filters='')
    tokenizer.fit_on_texts(train_df['Sequence'])
    
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df['Sequence']),
        maxlen=seq_length,
        padding='post',
        truncating='post'
    )
    X_test = pad_sequences(
        tokenizer.texts_to_sequences(test_df['Sequence']),
        maxlen=seq_length,
        padding='post',
        truncating='post'
    )
    
    y_train = train_df['Class'].values
    y_test = test_df['Class'].values
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    
    # Model configuration
    model = configure_model(
        vocab_size=len(tokenizer.word_index) + 1,
        seq_length=seq_length,
        num_classes=9
    )
    
    # Training callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]
    
    # Model training
    history = model.fit(
        X_train, y_train,
        batch_size=256,
        epochs=200,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Generate predictions
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    
    # Save outputs
    save_model_and_reports(model, history, y_test, y_pred, y_probs, output_dir)
    
    # Save tokenizer for inference
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "path/to/your/directory"
    OUTPUT_DIR = "path/to/your/directory"
    SEQ_LENGTH = 400  # Optimal for protein sequences
    
    # Run training pipeline
    train_model(
        training_path=os.path.join(INPUT_DIR, "training.csv"),
        validation_path=os.path.join(INPUT_DIR, "test.csv"),
        output_dir=OUTPUT_DIR,
        seq_length=SEQ_LENGTH
    )

