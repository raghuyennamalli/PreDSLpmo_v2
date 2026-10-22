#!/usr/bin/env python3

# MULTICLASS ENSEMBLE FEATURE SELECTION

import os
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import kruskal

from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


INPUT_FILE = (
    "/home/ragothaman/Sweety/LPMO/"
    "New_Version/Cluster_Aware_Spliting_Changed/"
    "Training_Validation_Final/Feature_Extraction/"
    "combined_dataset/multiclass_training/"
    "PredLPMO_multiclass_training_clean.csv"
)


OUTPUT_DIR = (
    "/home/ragothaman/Sweety/LPMO/"
    "New_Version/Cluster_Aware_Spliting_Changed/"
    "Training_Validation_Final/Feature_Extraction/"
    "EFS/multiclass_results"
)


ID_COLUMN = "Sequence ID_#"
FAMILY_COLUMN = "Family"
CLASS_COLUMN = "Class"

RANDOM_STATE = 42

RF_N_ESTIMATORS = 200
RF_N_JOBS = -1

LR_MAX_ITER = 1000

MI_RANDOM_STATE = 42

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 80)
print("MULTICLASS ENSEMBLE FEATURE SELECTION")
print("=" * 80)

print("\nClassification:")
print("9-class LPMO / Non-LPMO classification")

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_DIR)


if not os.path.exists(INPUT_FILE):

    print(
        "\nERROR: Input file does not exist:"
    )

    print(INPUT_FILE)

    sys.exit(1)


print("\n" + "=" * 80)
print("READING MULTICLASS TRAINING DATA")
print("=" * 80)

print("\nReading dataset...")

df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nRows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


required_columns = [
    ID_COLUMN,
    FAMILY_COLUMN,
    CLASS_COLUMN
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    print(
        "\nERROR: Required columns missing:"
    )

    for col in missing_columns:

        print(
            f"  {col}"
        )

    sys.exit(1)


print(
    "\nRequired columns: PASS"
)


duplicate_ids = (
    df[ID_COLUMN]
    .duplicated()
    .sum()
)


print(
    f"\nDuplicate Sequence IDs: "
    f"{duplicate_ids}"
)


if duplicate_ids != 0:

    print(
        "\nERROR: Duplicate Sequence IDs detected."
    )

    sys.exit(1)


print(
    "Sequence IDs: PASS"
)


print("\n" + "=" * 80)
print("MULTICLASS DISTRIBUTION")
print("=" * 80)


distribution = (
    df.groupby(
        [
            FAMILY_COLUMN,
            CLASS_COLUMN
        ]
    )
    .size()
    .reset_index(
        name="Sequence_Count"
    )
    .sort_values(
        CLASS_COLUMN
    )
)


print(
    distribution.to_string(
        index=False
    )
)


classes = sorted(
    df[CLASS_COLUMN]
    .dropna()
    .unique()
)


print(
    "\nClasses detected:"
)

print(
    classes
)


expected_classes = list(
    range(9)
)


if classes != expected_classes:

    print(
        "\nERROR: Expected classes:"
    )

    print(
        expected_classes
    )

    print(
        "\nDetected:"
    )

    print(
        classes
    )

    sys.exit(1)


print(
    "Nine-class structure: PASS"
)


class_per_id = (
    df.groupby(
        ID_COLUMN
    )[CLASS_COLUMN]
    .nunique()
)


if class_per_id.max() != 1:

    print(
        "\nERROR: Some Sequence IDs have "
        "multiple classes."
    )

    sys.exit(1)


print(
    "One class per sequence: PASS"
)


metadata_columns = [
    ID_COLUMN,
    FAMILY_COLUMN,
    CLASS_COLUMN
]


feature_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]


print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)


print(
    f"\nNumber of feature columns : "
    f"{len(feature_columns)}"
)


non_numeric = [
    col
    for col in feature_columns
    if not pd.api.types.is_numeric_dtype(
        df[col]
    )
]


if non_numeric:

    print(
        "\nERROR: Non-numeric features detected:"
    )

    for col in non_numeric[:20]:

        print(
            f"  {col}"
        )

    sys.exit(1)


print(
    "Numeric features: PASS"
)


missing_values = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)


print(
    f"\nMissing feature values: "
    f"{missing_values}"
)


if missing_values != 0:

    print(
        "\nERROR: Missing feature values "
        "detected."
    )

    sys.exit(1)


print(
    "Missing values: PASS"
)

nunique_features = (
    df[feature_columns]
    .nunique()
)


constant_features = (
    nunique_features[
        nunique_features <= 1
    ]
    .index
    .tolist()
)


print(
    f"\nConstant features: "
    f"{len(constant_features)}"
)


if constant_features:

    print(
        "\nNOTE: Constant features will receive "
        "zero importance."
    )


print("\n" + "=" * 80)
print("PREPARING FEATURE MATRIX")
print("=" * 80)


X = (
    df[feature_columns]
    .astype(
        np.float64
    )
)


y = (
    df[CLASS_COLUMN]
    .astype(int)
)


print(
    f"\nX shape : {X.shape}"
)


print(
    f"y shape : {y.shape}"
)


X_array = X.values
y_array = y.values



def minmax_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    minimum = np.min(
        values
    )

    maximum = np.max(
        values
    )

    if maximum == minimum:

        return np.zeros_like(
            values,
            dtype=np.float64
        )

    return (
        (values - minimum)
        /
        (maximum - minimum)
    )


print("\n" + "=" * 80)
print("METHOD 1/5: KRUSKAL-WALLIS H-TEST")
print("=" * 80)


print(
    "\nCalculating Kruskal-Wallis "
    "statistics for all features..."
)


kw_scores = np.zeros(
    len(feature_columns),
    dtype=np.float64
)

class_masks = {
    c: (y_array == c)
    for c in classes
}


for i, feature in enumerate(
    feature_columns
):

    values = X_array[:, i]

    groups = [
        values[class_masks[c]]
        for c in classes
    ]

    try:

        statistic, p_value = kruskal(
            *groups
        )

        if np.isfinite(
            statistic
        ):

            kw_scores[i] = statistic

        else:

            kw_scores[i] = 0.0

    except Exception:

        kw_scores[i] = 0.0

    if (i + 1) % 1000 == 0:

        print(
            f"Processed {i + 1}/"
            f"{len(feature_columns)} features"
        )


print(
    "\nKruskal-Wallis calculation complete."
)


kw_normalized = minmax_normalize(
    kw_scores
)


print("\n" + "=" * 80)
print("METHOD 2/5: ANOVA F-STATISTIC")
print("=" * 80)


print(
    "\nCalculating ANOVA F-statistics..."
)


f_scores, f_pvalues = f_classif(
    X_array,
    y_array
)


f_scores = np.nan_to_num(
    f_scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


anova_normalized = minmax_normalize(
    f_scores
)


print(
    "ANOVA calculation complete."
)


print("\n" + "=" * 80)
print("METHOD 3/5: MUTUAL INFORMATION")
print("=" * 80)


print(
    "\nCalculating mutual information..."
)


mi_scores = mutual_info_classif(
    X_array,
    y_array,
    random_state=MI_RANDOM_STATE
)


mi_scores = np.nan_to_num(
    mi_scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


mi_normalized = minmax_normalize(
    mi_scores
)


print(
    "Mutual information calculation complete."
)


print("\n" + "=" * 80)
print("METHOD 4/5: MULTINOMIAL LOGISTIC REGRESSION")
print("=" * 80)


print(
    "\nStandardizing features for "
    "multinomial logistic regression..."
)


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X_array
)


print(
    "Training multinomial logistic "
    "regression..."
)


lr_model = LogisticRegression(
    max_iter=LR_MAX_ITER,
    multi_class="multinomial",
    solver="lbfgs",
    random_state=RANDOM_STATE
)


lr_model.fit(
    X_scaled,
    y_array
)

lr_coefficients = np.abs(
    lr_model.coef_
)


lr_scores = np.mean(
    lr_coefficients,
    axis=0
)


lr_scores = np.nan_to_num(
    lr_scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


lr_normalized = minmax_normalize(
    lr_scores
)


print(
    "Multinomial logistic regression "
    "calculation complete."
)


print("\n" + "=" * 80)
print("METHOD 5A/6: RANDOM FOREST - GINI")
print("=" * 80)


print(
    f"\nTraining Random Forest "
    f"({RF_N_ESTIMATORS} trees)..."
)


rf_gini = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    criterion="gini",
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS,
    class_weight=None
)


rf_gini.fit(
    X_array,
    y_array
)


rf_gini_scores = (
    rf_gini.feature_importances_
)


rf_gini_scores = np.nan_to_num(
    rf_gini_scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


rf_gini_normalized = minmax_normalize(
    rf_gini_scores
)


print(
    "Random Forest Gini calculation complete."
)


print("\n" + "=" * 80)
print("METHOD 5B/6: RANDOM FOREST - ENTROPY")
print("=" * 80)


print(
    f"\nTraining Random Forest "
    f"({RF_N_ESTIMATORS} trees)..."
)


rf_entropy = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    criterion="entropy",
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS,
    class_weight=None
)


rf_entropy.fit(
    X_array,
    y_array
)


rf_entropy_scores = (
    rf_entropy.feature_importances_
)


rf_entropy_scores = np.nan_to_num(
    rf_entropy_scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


rf_entropy_normalized = minmax_normalize(
    rf_entropy_scores
)


print(
    "Random Forest entropy calculation complete."
)


print("\n" + "=" * 80)
print("CALCULATING FINAL EFS SCORES")
print("=" * 80)


results = pd.DataFrame({

    "Feature":
        feature_columns,

    "Kruskal_Wallis":
        kw_normalized,

    "ANOVA_F":
        anova_normalized,

    "Mutual_Information":
        mi_normalized,

    "Multinomial_Logistic_Regression":
        lr_normalized,

    "RF_Gini":
        rf_gini_normalized,

    "RF_Entropy":
        rf_entropy_normalized
})


score_columns = [

    "Kruskal_Wallis",

    "ANOVA_F",

    "Mutual_Information",

    "Multinomial_Logistic_Regression",

    "RF_Gini",

    "RF_Entropy"
]


results["EFS_Score"] = (
    results[score_columns]
    .mean(axis=1)
)


results = (
    results
    .sort_values(
        "EFS_Score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


results.insert(
    0,
    "Rank",
    np.arange(
        1,
        len(results) + 1
    )
)


complete_file = os.path.join(
    OUTPUT_DIR,
    "multiclass_EFS_complete_ranking.csv"
)


results.to_csv(
    complete_file,
    index=False
)


print(
    "\nComplete ranking saved:"
)


print(
    complete_file
)


threshold_05 = results[
    results["EFS_Score"] > 0.5
].copy()


threshold_05_file = os.path.join(
    OUTPUT_DIR,
    "multiclass_EFS_gt_0.5.csv"
)


threshold_05.to_csv(
    threshold_05_file,
    index=False
)


print(
    f"\nFeatures with EFS_Score > 0.5:"
    f" {len(threshold_05)}"
)


print(
    threshold_05_file
)


threshold_06 = results[
    results["EFS_Score"] > 0.6
].copy()


threshold_06_file = os.path.join(
    OUTPUT_DIR,
    "multiclass_EFS_gt_0.6.csv"
)


threshold_06.to_csv(
    threshold_06_file,
    index=False
)


print(
    f"\nFeatures with EFS_Score > 0.6:"
    f" {len(threshold_06)}"
)


print(
    threshold_06_file
)


feature_list_05 = os.path.join(
    OUTPUT_DIR,
    "multiclass_features_gt_0.5.txt"
)


with open(
    feature_list_05,
    "w"
) as f:

    for feature in threshold_05[
        "Feature"
    ]:

        f.write(
            feature + "\n"
        )


feature_list_06 = os.path.join(
    OUTPUT_DIR,
    "multiclass_features_gt_0.6.txt"
)


with open(
    feature_list_06,
    "w"
) as f:

    for feature in threshold_06[
        "Feature"
    ]:

        f.write(
            feature + "\n"
        )


summary_file = os.path.join(
    OUTPUT_DIR,
    "multiclass_EFS_summary.csv"
)


summary = pd.DataFrame([{

    "Classes":
        len(classes),

    "Original_Features":
        len(feature_columns),

    "EFS_gt_0.5":
        len(threshold_05),

    "EFS_gt_0.6":
        len(threshold_06),

    "Top_EFS_Score":
        results["EFS_Score"].max(),

    "Median_EFS_Score":
        results["EFS_Score"].median(),

    "Minimum_EFS_Score":
        results["EFS_Score"].min()

}])


summary.to_csv(
    summary_file,
    index=False
)



print("\n" + "=" * 80)
print("TOP 20 FEATURES")
print("=" * 80)


print(
    results[
        [
            "Rank",
            "Feature",
            "EFS_Score"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("MULTICLASS EFS COMPLETE")
print("=" * 80)


print(
    f"\nNumber of classes       : "
    f"{len(classes)}"
)


print(
    f"Original features      : "
    f"{len(feature_columns)}"
)


print(
    f"EFS > 0.5 features     : "
    f"{len(threshold_05)}"
)


print(
    f"EFS > 0.6 features     : "
    f"{len(threshold_06)}"
)


print("\nOutput files:")


print(
    f"1. {complete_file}"
)


print(
    f"2. {threshold_05_file}"
)


print(
    f"3. {threshold_06_file}"
)


print(
    f"4. {feature_list_05}"
)


print(
    f"5. {feature_list_06}"
)


print(
    f"6. {summary_file}"
)



