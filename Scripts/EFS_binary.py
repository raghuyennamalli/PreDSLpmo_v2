#!/usr/bin/env python3

# BINARY ENSEMBLE FEATURE SELECTION (EFS)

import os
import glob
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


BASE_DIR = (
    "/home/ragothaman/Sweety/LPMO/"
    "New_Version/Cluster_Aware_Spliting_Changed/"
    "Training_Validation_Final/Feature_Extraction/"
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "EFS",
    "binary_datasets"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "EFS",
    "binary_results"
)

RANDOM_STATE = 42

RF_TREES = 300


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 80)
print("BINARY ENSEMBLE FEATURE SELECTION")
print("=" * 80)

print("\nClassification strategy:")
print("LPMO family vs Non-LPMO")

print("\nInput:")
print(INPUT_DIR)

print("\nOutput:")
print(OUTPUT_DIR)

print("\nRandom state:")
print(RANDOM_STATE)

print("\nRandom Forest trees:")
print(RF_TREES)


def normalize_scores(scores):

    scores = np.asarray(scores, dtype=float)

    scores[~np.isfinite(scores)] = 0.0

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum == minimum:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)

def perform_efs(input_file):

    print("\n" + "=" * 80)
    print("PROCESSING")
    print("=" * 80)

    print(input_file)

    df = pd.read_csv(input_file)

    id_column = "Sequence ID_#"
    target_column = "Binary_Class"


    if id_column not in df.columns:
        raise ValueError(
            f"{id_column} missing from {input_file}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"{target_column} missing from {input_file}"
        )


    feature_columns = [
        c for c in df.columns
        if c not in [
            id_column,
            target_column
        ]
    ]


    if len(feature_columns) == 0:
        raise ValueError(
            f"No feature columns found in {input_file}"
        )

    non_numeric = []

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric.append(column)

    if non_numeric:

        raise ValueError(
            "Non-numeric feature columns found:\n"
            + "\n".join(non_numeric[:20])
        )

    missing_values = df[feature_columns].isna().sum().sum()

    if missing_values > 0:

        raise ValueError(
            f"Missing feature values detected: {missing_values}"
        )


    unique_classes = sorted(df[target_column].unique().tolist())

    if unique_classes != [0, 1]:

        raise ValueError(
            f"Binary_Class must contain exactly 0 and 1. "
            f"Found: {unique_classes}"
        )

    X = df[feature_columns].values
    y = df[target_column].values

    print("\nDataset:")
    print(f"Sequences : {len(df)}")
    print(f"Features  : {len(feature_columns)}")
    print(f"Positive  : {(y == 1).sum()}")
    print(f"Negative  : {(y == 0).sum()}")


    print("\nBinary classification:")
    print("Positive class = LPMO family")
    print("Negative class = Non-LPMO")


    scores = pd.DataFrame(
        index=feature_columns
    )


    print("\n[1/6] Mann-Whitney U test")

    X_positive = X[y == 1]
    X_negative = X[y == 0]

    mwu_scores = []

    for i in range(X.shape[1]):

        try:

            positive_values = X_positive[:, i]
            negative_values = X_negative[:, i]

            # Mann-Whitney U test
            _, p = mannwhitneyu(
                positive_values,
                negative_values,
                alternative="two-sided"
            )

            median_difference = abs(
                np.median(positive_values)
                -
                np.median(negative_values)
            )

            if p <= 0:
                significance = 300.0
            else:
                significance = -np.log10(p)

            score = significance * (
                median_difference + 1e-12
            )

            mwu_scores.append(score)

        except Exception:
            mwu_scores.append(0.0)

    scores["Mann_Whitney"] = normalize_scores(
        mwu_scores
    )

    print("[2/6] Pearson correlation")

    pearson_scores = []

    for i in range(X.shape[1]):

        try:

            r, _ = pearsonr(
                X[:, i],
                y
            )

            pearson_scores.append(
                abs(r)
            )

        except Exception:

            pearson_scores.append(0.0)

    scores["Pearson"] = normalize_scores(
        pearson_scores
    )

    print("[3/6] Spearman correlation")

    spearman_scores = []

    for i in range(X.shape[1]):

        try:

            r, _ = spearmanr(
                X[:, i],
                y
            )

            if np.isfinite(r):

                spearman_scores.append(
                    abs(r)
                )

            else:

                spearman_scores.append(0.0)

        except Exception:

            spearman_scores.append(0.0)

    scores["Spearman"] = normalize_scores(
        spearman_scores
    )

    print("[4/6] Logistic Regression")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    logistic = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    logistic.fit(
        X_scaled,
        y
    )

    logistic_scores = np.abs(
        logistic.coef_[0]
    )

    scores["Logistic_Regression"] = normalize_scores(
        logistic_scores
    )

    print("[5/6] Random Forest - Gini")

    rf_gini = RandomForestClassifier(
        n_estimators=RF_TREES,
        criterion="gini",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    rf_gini.fit(
        X,
        y
    )

    gini_scores = rf_gini.feature_importances_

    scores["RF_Gini"] = normalize_scores(
        gini_scores
    )

    print("[6/6] Random Forest - Entropy")

    rf_entropy = RandomForestClassifier(
        n_estimators=RF_TREES,
        criterion="entropy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    rf_entropy.fit(
        X,
        y
    )

    entropy_scores = rf_entropy.feature_importances_

    scores["RF_Entropy"] = normalize_scores(
        entropy_scores
    )

    print("\nCombining six feature-selection methods...")

    method_columns = [
        "Mann_Whitney",
        "Pearson",
        "Spearman",
        "Logistic_Regression",
        "RF_Gini",
        "RF_Entropy"
    ]

    scores["EFS_Score"] = scores[
        method_columns
    ].mean(axis=1)

    scores.index.name = "Feature"

    scores = scores.sort_values(
        "EFS_Score",
        ascending=False
    )

    basename = os.path.basename(
        input_file
    )

    family = basename.replace(
        "EFS_",
        ""
    ).replace(
        "_vs_NonLPMO_training.csv",
        ""
    )

    family_dir = os.path.join(
        OUTPUT_DIR,
        family
    )

    os.makedirs(
        family_dir,
        exist_ok=True
    )

    ranking_file = os.path.join(
        family_dir,
        f"{family}_EFS_complete_ranking.csv"
    )

    scores.to_csv(
        ranking_file
    )

    selected_05 = scores[
        scores["EFS_Score"] > 0.5
    ]

    selected_05_features = (
        selected_05.index.tolist()
    )

    selected_06 = scores[
        scores["EFS_Score"] > 0.6
    ]

    selected_06_features = (
        selected_06.index.tolist()
    )

    output_05 = df[
        [id_column]
        +
        selected_05_features
        +
        [target_column]
    ].copy()

    file_05 = os.path.join(
        family_dir,
        f"{family}_EFS_gt_0.5.csv"
    )

    output_05.to_csv(
        file_05,
        index=False
    )

    output_06 = df[
        [id_column]
        +
        selected_06_features
        +
        [target_column]
    ].copy()

    file_06 = os.path.join(
        family_dir,
        f"{family}_EFS_gt_0.6.csv"
    )

    output_06.to_csv(
        file_06,
        index=False
    )

    print("\n" + "-" * 80)
    print(f"{family} EFS SUMMARY")
    print("-" * 80)

    print(
        f"Original features : {len(feature_columns)}"
    )

    print(
        f"EFS > 0.5         : {len(selected_05_features)}"
    )

    print(
        f"EFS > 0.6         : {len(selected_06_features)}"
    )

    print(
        f"Ranking saved      : {ranking_file}"
    )

    print(
        f">0.5 dataset       : {file_05}"
    )

    print(
        f">0.6 dataset       : {file_06}"
    )


    return {
        "Family": family,
        "Original_Features": len(feature_columns),
        "EFS_gt_0.5": len(selected_05_features),
        "EFS_gt_0.6": len(selected_06_features)
    }

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "EFS_*_vs_NonLPMO_training.csv"
        )
    )
)


if not files:

    print("\nERROR: No binary datasets found.")

    print("\nExpected files:")
    print(
        "EFS_AA9_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA10_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA11_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA13_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA14_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA15_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA16_vs_NonLPMO_training.csv"
    )
    print(
        "EFS_AA17_vs_NonLPMO_training.csv"
    )

    sys.exit(1)


print(
    f"\nBinary datasets detected: {len(files)}"
)

for file in files:

    print(
        "  ",
        os.path.basename(file)
    )


all_summary = []

for file in files:

    result = perform_efs(
        file
    )

    all_summary.append(
        result
    )


summary_df = pd.DataFrame(
    all_summary
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "binary_EFS_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 80)
print("BINARY EFS COMPLETE")
print("=" * 80)

print("\nFeature-selection summary:")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nOutput directory:")
print(OUTPUT_DIR)

