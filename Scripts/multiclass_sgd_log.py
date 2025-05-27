#!/usr/bin/env python3
# Authors: Vaishnavi Saravanan
# Version: 1.0
# Last Modified: 2025-04-25
# Description:
#   Trains a multiclass SGD classifier with log loss using grid search for hyperparameter tuning.
#   Features enhanced class weighting, early stopping, elastic net penalty, and automated visualization.
#   Saves model artifacts, evaluation metrics, and diagnostic plots.
#
# Usage:
#   python multiclass_sgd_log.py
#   (Ensure training_multiclass.csv and validation_multiclass.csv exist in base_path)
#
# Input:
#   - training_multiclass.csv: Training data with class labels in first column
#   - validation_multiclass.csv: Validation data with same format
# Output:
#   - Saved model (multiclass_model.joblib)
#   - Preprocessing artifacts (imputer.joblib, scaler.joblib)
#   - Evaluation reports (classification_report.txt)
#   - Diagnostic plots (confusion_matrix.png, pr_curve.png, roc_curve.png, tuning_heatmap.png)

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib

# Suppress warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure paths
base_path = "path/to/your/directory"
result_folder = "log_loss"
output_dir = os.path.join(base_path, result_folder)
os.makedirs(output_dir, exist_ok=True)

def saveBestModel(classifier, X_test, Y_test):
    print("\n Saving the best model...")
    model_path = os.path.join(output_dir, "multiclass_model.joblib")
    joblib.dump(classifier, model_path)

    Y_pred = classifier.predict(X_test)
    classes = sorted(classifier.classes_)

    # Classification Report
    print("\n Classification Report:")
    report = classification_report(Y_test, Y_pred, labels=classes, zero_division=0)
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(12,10))
    cm = confusion_matrix(Y_test, Y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    # ROC/PR Curves
    Y_score = classifier.decision_function(X_test)
    Y_test_bin = label_binarize(Y_test, classes=classes)

    # Precision-Recall Curve
    plt.figure(figsize=(10,8))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(Y_test_bin[:,i], Y_score[:,i])
        plt.plot(recall, precision, lw=2, label=f'Class {classes[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), bbox_inches='tight')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(10,8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(Y_test_bin[:,i], Y_score[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), bbox_inches='tight')
    plt.close()

def plot_cv_heatmap(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    params = results['params'].apply(lambda x: pd.Series(x))
    results = pd.concat([params, results['mean_test_score']], axis=1)
    
    # Handle duplicate parameter combinations
    heatmap_data = results.groupby(['alpha', 'max_iter'])['mean_test_score'].mean().unstack()
    
    plt.figure(figsize=(12,8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
                    cbar_kws={'label':'Accuracy'})
    ax.set_yticklabels([f"1e{int(np.log10(float(t.get_text())))}" 
                      for t in ax.get_yticklabels()], rotation=0)
    plt.title('Hyperparameter Tuning')
    plt.xlabel('Max Iterations')
    plt.ylabel('Alpha (log scale)')
    plt.savefig(os.path.join(output_dir, "tuning_heatmap.png"), bbox_inches='tight')
    plt.close()

def getBestModel(X_train, X_test, Y_train, Y_test):
    # Enhanced class weighting
    class_counts = np.bincount(Y_train)
    total_samples = len(Y_train)
    class_weights = {i: total_samples/(len(class_counts)*count) for i, count in enumerate(class_counts)}
    
    param_grid = {
        'alpha': [10**x for x in range(-7,1)],
        'max_iter': [1000, 2000],  
        'class_weight': [class_weights, 'balanced'],
        'tol': [1e-4] 
    }

    model = SGDClassifier(
        loss="log_loss",
        penalty='elasticnet',
        early_stopping=True,
        learning_rate='adaptive',
        eta0=0.1
    )

    print("\n\ Starting GridSearchCV with enhanced class weighting")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, Y_train)
    plot_cv_heatmap(grid_search)
    best_model = grid_search.best_estimator_
    print(f"\n Best params: {grid_search.best_params_}")
    saveBestModel(best_model, X_test, Y_test)
    return best_model

def main():
    train_path = os.path.join(base_path, "training_multiclass.csv")
    test_path = os.path.join(base_path, "validation_multiclass.csv")

    print("\n\ Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("\n Preprocessing...")
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_df.iloc[:,1:])
    Y_train = train_df.iloc[:,0].values.astype(int)
    X_train = scaler.fit_transform(X_train)

    X_test = imputer.transform(test_df.iloc[:,1:])
    Y_test = test_df.iloc[:,0].values.astype(int)
    X_test = scaler.transform(X_test)

    joblib.dump(imputer, os.path.join(output_dir, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    print("\n Training with enhanced class weighting...")
    start = time.time()
    best_model = getBestModel(X_train, X_test, Y_train, Y_test)
    print(f"\n Total time: {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
