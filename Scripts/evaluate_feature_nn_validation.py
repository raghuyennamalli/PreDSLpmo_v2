import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)


BASE_DIR = (
    "/media/ragothaman/yennamalli/Sweety/LPMO_NEW/"
    "New_Version/Cluster_Aware_Spliting_Changed/"
    "Training_Validation_Final/Feature_Extraction"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "EFS/validation_selected_datasets/multiclass/"
    "PredLPMO_multiclass_validation_EFS_gt_0.5.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "Cluster_Aware_CV/Feature_NN_FINAL/feature_nn_final.keras"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "Cluster_Aware_CV/Feature_NN_FINAL/feature_nn_scaler.joblib"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "Cluster_Aware_CV/Feature_NN_FINAL/feature_nn_metadata.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Cluster_Aware_CV/Feature_NN_FINAL/validation_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


CLASS_NAMES = [
    "AA9",
    "AA10",
    "AA11",
    "AA13",
    "AA14",
    "AA15",
    "AA16",
    "AA17",
    "Non-LPMO"
]

LABEL_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

INDEX_TO_LABEL = {
    index: name
    for index, name in enumerate(CLASS_NAMES)
}

required_files = [
    VALIDATION_FILE,
    MODEL_FILE,
    SCALER_FILE,
    METADATA_FILE
]

for file_path in required_files:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


print("Loading validation dataset...")

validation = pd.read_csv(VALIDATION_FILE)

print(f"Total validation sequences: {len(validation)}")

if validation.empty:
    raise ValueError("The validation dataset is empty.")


id_candidates = [
    "Sequence_ID",
    "sequence_id",
    "ID",
    "id",
    "Protein_ID",
    "protein_id",
    "Accession",
    "accession",
    "Sequence",
    "sequence"
]

id_col = next(
    (
        column for column in id_candidates
        if column in validation.columns
    ),
    None
)

if id_col is not None:
    duplicate_count = int(
        validation[id_col].duplicated().sum()
    )

    if duplicate_count:
        print(
            f"WARNING: Found {duplicate_count} duplicate "
            f"values in '{id_col}'. Retaining all records."
        )
    else:
        print("No duplicate sequence IDs detected.")
else:
    print(
        "WARNING: No recognized sequence ID column found. "
    )


with open(METADATA_FILE, "r") as file:
    metadata = json.load(file)

feature_names = metadata.get("feature_names_in_order")

if not feature_names:
    raise ValueError(
        "The metadata file does not contain "
        "'feature_names_in_order'."
    )

print(f"Number of selected features: {len(feature_names)}")


label_candidates = [
    "Class",
    "class",
    "Label",
    "label",
    "Family",
    "family",
    "True_Label",
    "true_label"
]

label_col = next(
    (
        column for column in label_candidates
        if column in validation.columns
    ),
    None
)

if label_col is None:
    raise ValueError(
        "Could not identify the true-label column. "
        f"Available columns: {validation.columns.tolist()}"
    )

print(f"True-label column: {label_col}")


raw_labels = validation[label_col].astype(str).str.strip()

numeric_to_class = {
    str(index): name
    for index, name in enumerate(CLASS_NAMES)
}

if set(raw_labels).issubset(set(numeric_to_class.keys())):

    # Validation labels are numeric strings: 0, 1, ..., 8
    y_true = raw_labels.astype(int).to_numpy()

    y_labels = raw_labels.map(numeric_to_class)

    print("Numeric class labels detected.")

else:

    unknown_labels = sorted(
        set(raw_labels) - set(CLASS_NAMES)
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected validation labels: {unknown_labels}\n"
            f"Expected class names: {CLASS_NAMES}\n"
            "Alternatively, labels can be numeric values 0-8."
        )

    y_labels = raw_labels

    y_true = y_labels.map(LABEL_TO_INDEX).to_numpy()

print("\nTrue class distribution:")

print(
    y_labels.value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .to_string()
)


missing_features = [
    feature
    for feature in feature_names
    if feature not in validation.columns
]

if missing_features:
    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )

X = validation[feature_names].apply(
    pd.to_numeric,
    errors="coerce"
)

if X.isnull().any().any():

    missing_count = int(X.isnull().sum().sum())

    raise ValueError(
        f"Found {missing_count} missing or non-numeric "
        "feature values. Check the validation CSV."
    )

if not np.isfinite(X.to_numpy()).all():
    raise ValueError(
        "The validation features contain infinite values."
    )


print("\nLoading trained Feature Neural Network...")

model = tf.keras.models.load_model(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)

print("Model loaded successfully.")
print("Scaler loaded successfully.")


print("\nScaling validation features...")

X_scaled = scaler.transform(X)

if not np.isfinite(X_scaled).all():
    raise ValueError(
        "Scaled validation features contain non-finite values."
    )


print("\nPredicting validation classes...")

probabilities = model.predict(
    X_scaled,
    batch_size=256,
    verbose=1
)

if probabilities.ndim != 2:
    raise ValueError(
        f"Unexpected prediction shape: {probabilities.shape}"
    )

if probabilities.shape[1] != len(CLASS_NAMES):
    raise ValueError(
        f"The model produces {probabilities.shape[1]} outputs, "
        f"but {len(CLASS_NAMES)} classes were expected."
    )

y_pred = np.argmax(probabilities, axis=1)

predicted_labels = [
    INDEX_TO_LABEL[index]
    for index in y_pred
]


predictions = validation.copy()

predictions["True_Label"] = y_labels.to_numpy()

predictions["Predicted_Label"] = predicted_labels

predictions["Correct_Prediction"] = (
    predictions["True_Label"]
    == predictions["Predicted_Label"]
)

for index, class_name in enumerate(CLASS_NAMES):

    predictions[f"Probability_{class_name}"] = (
        probabilities[:, index]
    )

prediction_file = os.path.join(
    OUTPUT_DIR,
    "validation_predictions.csv"
)

predictions.to_csv(
    prediction_file,
    index=False
)

print(f"\nPredictions saved:\n{prediction_file}")


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{name}"
        for name in CLASS_NAMES
    ],
    columns=[
        f"Predicted_{name}"
        for name in CLASS_NAMES
    ]
)

cm_file = os.path.join(
    OUTPUT_DIR,
    "validation_confusion_matrix.csv"
)

cm_df.to_csv(cm_file)

print(f"\nConfusion matrix saved:\n{cm_file}")


precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0
    )
)

total = len(y_true)

metrics_rows = []

for index, class_name in enumerate(CLASS_NAMES):

    
    TP = int(cm[index, index])

    FN = int(cm[index, :].sum() - TP)

    FP = int(cm[:, index].sum() - TP)

    TN = int(total - TP - FN - FP)

    class_precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )

    sensitivity = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )

    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else 0.0
    )

    # F1 = 2 * Precision * Recall / (Precision + Recall)
    f1_score = (
        2 * class_precision * sensitivity
        / (class_precision + sensitivity)
        if (class_precision + sensitivity) > 0
        else 0.0
    )

    metrics_rows.append({
        "Class": class_name,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": class_precision,
        "Recall": sensitivity,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1_Score": f1_score,
        "Support": int(support[index])
    })

metrics_df = pd.DataFrame(metrics_rows)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "validation_per_class_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print(f"\nPer-class metrics saved:\n{metrics_file}")


accuracy = accuracy_score(y_true, y_pred)

macro_precision = float(np.mean(precision))
macro_recall = float(np.mean(recall))
macro_f1 = float(np.mean(f1))

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)

summary = {
    "Total_Sequences": total,
    "Accuracy": accuracy,
    "Macro_Precision": macro_precision,
    "Macro_Recall": macro_recall,
    "Macro_F1_Score": macro_f1,
    "Weighted_Precision": float(weighted_precision),
    "Weighted_Recall": float(weighted_recall),
    "Weighted_F1_Score": float(weighted_f1)
}

summary_file = os.path.join(
    OUTPUT_DIR,
    "validation_summary_metrics.csv"
)

pd.DataFrame([summary]).to_csv(
    summary_file,
    index=False
)


report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_file = os.path.join(
    OUTPUT_DIR,
    "validation_classification_report.csv"
)

report_df.to_csv(report_file)


print("\n" + "=" * 65)
print("INDEPENDENT VALIDATION RESULTS")
print("=" * 65)

print(f"Total validation sequences : {total}")
print(f"Accuracy                   : {accuracy:.4f}")
print(f"Macro Precision            : {macro_precision:.4f}")
print(f"Macro Recall               : {macro_recall:.4f}")
print(f"Macro F1-score             : {macro_f1:.4f}")
print(f"Weighted Precision         : {weighted_precision:.4f}")
print(f"Weighted Recall            : {weighted_recall:.4f}")
print(f"Weighted F1-score          : {weighted_f1:.4f}")

print("\nPER-CLASS METRICS")
print("-" * 65)

print(
    metrics_df.round(4).to_string(index=False)
)

print("\nCONFUSION MATRIX")
print("-" * 65)

print(cm_df)

print("\n" + "=" * 65)
print("ALL RESULTS SAVED SUCCESSFULLY")
print("=" * 65)

print(f"\nOutput directory:\n{OUTPUT_DIR}")

