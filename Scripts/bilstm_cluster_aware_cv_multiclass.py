#!/usr/bin/env python3
"""
CLUSTER-AWARE 5-FOLD CV FOR MULTICLASS Bi-LSTM  
Usage
-----
    python bilstm_cluster_cv_final.py --smoke_test        # quick full-pipeline test
    python bilstm_cluster_cv_final.py                     # real run: 5 folds + aggregate + final
    python bilstm_cluster_cv_final.py --fold 3            # one fold only (e.g. parallel jobs)
    python bilstm_cluster_cv_final.py --aggregate_only    # after all 5 folds exist on disk
    python bilstm_cluster_cv_final.py --skip_final        # CV only, no final model
"""


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TerminateOnNaN,
    Callback,
)



BASE = Path(
    "/media/ragothaman/yennamalli/Sweety/LPMO_NEW/New_Version/"
    "Cluster_Aware_Spliting_Changed/Training_Validation_Final"
)

TRAIN_FASTA_DIR = BASE / "Training" / "cleaned_fasta"

FOLD_FILE = (
    BASE
    / "Feature_Extraction"
    / "Cluster_Aware_CV"
    / "fold_assignments"
    / "multiclass_training_cluster_aware_5fold.csv"
)

OUTPUT_DIR = (
    BASE
    / "Feature_Extraction"
    / "Cluster_Aware_CV"
    / "BiLSTM_MULTICLASS_FINAL_V2"
)

SMOKE_DIR_NAME = "SMOKE_TEST"

RANDOM_STATE = 42

N_OUTER_FOLDS = 5

# ONE deterministic inner split (split 0 of a StratifiedGroupKFold).
INNER_SPLITTER_FOLDS = 5
INNER_SPLIT_INDEX = 0

NUM_CLASSES = 9
CLASS_LABELS = list(range(9))

SEQ_LENGTH = 400
BATCH_SIZE = 256
MAX_EPOCHS = 200

LEARNING_RATE = 0.0005
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-6

# Model size
EMBED_DIM = 256
LSTM1_UNITS = 512
LSTM2_UNITS = 256
DENSE_UNITS = 256

WEIGHTED_VAL_LOSS = True

SMOKE_PER_CLASS = 200
SMOKE_EPOCHS = 2

EXPECTED_N = 40016

EXPECTED_CLASS_COUNTS = {
    0: 8783,     # AA9
    1: 6751,     # AA10
    2: 2062,     # AA11
    3: 241,      # AA13
    4: 832,      # AA14
    5: 403,      # AA15
    6: 653,      # AA16
    7: 87,       # AA17
    8: 20204,    # Non-LPMO
}

CLASS_NAMES = {
    0: "AA9",
    1: "AA10",
    2: "AA11",
    3: "AA13",
    4: "AA14",
    5: "AA15",
    6: "AA16",
    7: "AA17",
    8: "Non-LPMO",
}


AMINO_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

VOCAB = {aa: i + 1 for i, aa in enumerate(AMINO_ALPHABET)}

UNK_INDEX = len(VOCAB) + 1

VOCAB_SIZE = len(VOCAB) + 2

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def build_encoder():
    """Plain-dict description of the encoding (safe to pickle anywhere)."""
    return {
        "vocab": dict(VOCAB),
        "pad_index": 0,
        "unk_index": int(UNK_INDEX),
        "vocab_size": int(VOCAB_SIZE),
        "seq_length": int(SEQ_LENGTH),
        "padding": "post",
        "truncating": "post",
    }


def encode_sequences(sequences, encoder=None):
    """Encode protein strings -> int32 array (n, seq_length), post pad/trunc."""
    enc = encoder if encoder is not None else build_encoder()

    lut = np.full(256, enc["unk_index"], dtype=np.int32)
    for aa, idx in enc["vocab"].items():
        lut[ord(aa)] = idx

    length = enc["seq_length"]
    X = np.zeros((len(sequences), length), dtype=np.int32)

    for i, seq in enumerate(sequences):
        raw = np.frombuffer(
            seq[:length].encode("latin-1", errors="replace"),
            dtype=np.uint8,
        )
        X[i, : len(raw)] = lut[raw]

    return X

def set_seed(seed):
    """Seeds Python, NumPy and TensorFlow. Call right before build_model()."""
    tf.keras.utils.set_random_seed(int(seed))


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def clean_history(history_dict):
    return {
        key: [float(v) for v in values]
        for key, values in history_dict.items()
    }


def summarize_schedule(lrs):
    if not lrs:
        return "(empty)"

    segments = []
    start = 0

    for i in range(1, len(lrs) + 1):
        if i == len(lrs) or lrs[i] != lrs[start]:
            segments.append(f"epochs {start + 1}-{i}: {lrs[start]:.10g}")
            start = i

    return "; ".join(segments)


def build_median_lr_schedule(fold_schedules, n_epochs):

    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")

    schedule = []

    for epoch_idx in range(n_epochs):
        available = [
            float(sched[epoch_idx])
            for sched in fold_schedules
            if epoch_idx < len(sched)
        ]

        if not available:
            raise RuntimeError(
                f"No fold LR schedule contains epoch {epoch_idx + 1}."
            )

        schedule.append(float(np.median(available)))

    return schedule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final cluster-aware multiclass Bi-LSTM CV"
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only one outer fold (1-5) and stop. Default: all five.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Maximum inner-training epochs. Default = 200.",
    )

    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help=(
            "Run the WHOLE pipeline (5 folds, refit, aggregation, final "
            "model) on ~1,700 sequences for 2 epochs. Writes to a separate "
            "SMOKE_TEST folder. Do NOT report smoke-test metrics."
        ),
    )

    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help=(
            "Do not train folds; aggregate the five saved fold outputs "
            "and train the final model."
        ),
    )

    parser.add_argument(
        "--skip_final",
        action="store_true",
        help="Skip training the final full-data model.",
    )

    args = parser.parse_args()

    if args.fold is not None and args.fold not in range(1, N_OUTER_FOLDS + 1):
        parser.error("--fold must be 1, 2, 3, 4 or 5.")

    if args.fold is not None and args.aggregate_only:
        parser.error("--fold and --aggregate_only cannot be combined.")

    return args


def normalize_id(value):
    return str(value).strip().lstrip(">").split()[0]


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_fold_assignments():
    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            f"\nFold file not found:\n{FOLD_FILE}\n\n"
            "The existing cluster-aware fold file must be present."
        )

    df = pd.read_csv(FOLD_FILE)

    id_col = find_column(
        df,
        ["Sequence_ID", "Sequence ID_#", "Sequence_ID_#", "ID"],
        "sequence ID",
    )
    class_col = find_column(df, ["Class", "class"], "class")
    fold_col = find_column(df, ["Fold", "fold"], "fold")
    cluster_col = find_column(
        df,
        ["Cluster_ID", "Cluster", "cluster_id", "cluster"],
        "cluster ID",
    )


    for col in (id_col, class_col, fold_col, cluster_col):
        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains {int(df[col].isna().sum())} "
                "missing value(s) in the fold file."
            )

    cluster_str = df[cluster_col].astype(str).str.strip()

    if (cluster_str == "").any() or (cluster_str.str.lower() == "nan").any():
        raise ValueError(
            "Empty or 'nan' cluster IDs found in the fold file. "
            "They would silently form one giant fake cluster."
        )

    out = pd.DataFrame(
        {
            "Sequence_ID": df[id_col].map(normalize_id),
            "Class": pd.to_numeric(df[class_col], errors="raise").astype(int),
            "Fold": pd.to_numeric(df[fold_col], errors="raise").astype(int),
            "Cluster_ID": cluster_str,
        }
    )

    if len(out) != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N:,} rows but found {len(out):,}."
        )

    if out["Sequence_ID"].duplicated().any():
        dup = out.loc[out["Sequence_ID"].duplicated(keep=False)]
        raise ValueError(
            "Duplicate sequence IDs found in fold assignment file.\n"
            f"{dup.head(20)}"
        )

    if (out.groupby("Sequence_ID")["Class"].nunique() > 1).any():
        raise ValueError("A sequence ID has multiple class labels.")

    if (out.groupby("Sequence_ID")["Cluster_ID"].nunique() > 1).any():
        raise ValueError("A sequence ID belongs to multiple clusters.")

    if sorted(out["Fold"].unique()) != [1, 2, 3, 4, 5]:
        raise ValueError(
            "Expected outer folds 1-5.\n"
            f"Found: {sorted(out['Fold'].unique())}"
        )

    if sorted(out["Class"].unique()) != CLASS_LABELS:
        raise ValueError(
            "Expected classes 0-8.\n"
            f"Found: {sorted(out['Class'].unique())}"
        )

    actual_counts = out["Class"].value_counts().sort_index().to_dict()

    if actual_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "\nClass counts do not match the expected dataset.\n"
            f"Expected: {EXPECTED_CLASS_COUNTS}\n"
            f"Actual:   {actual_counts}"
        )

    cluster_fold_counts = out.groupby("Cluster_ID")["Fold"].nunique()
    bad_clusters = cluster_fold_counts[cluster_fold_counts > 1]

    if len(bad_clusters) > 0:
        raise ValueError(
            "OUTER CLUSTER INTEGRITY FAILED.\n"
            "Clusters occur in multiple outer folds:\n"
            f"{bad_clusters.head(20)}"
        )

    print("\n" + "=" * 80)
    print("FOLD FILE AUDIT")
    print("=" * 80)
    print(f"Sequences : {len(out):,}")
    print(f"Unique IDs: {out['Sequence_ID'].nunique():,}")
    print(f"Clusters  : {out['Cluster_ID'].nunique():,}")
    print("Cluster integrity: PASS")

    print("\nClass counts:")
    print(out["Class"].value_counts().sort_index().to_string())

    print("\nOuter fold x class:")
    print(pd.crosstab(out["Fold"], out["Class"]).sort_index().to_string())

    return out


def _commit_record(records, seq_id, parts, path):
    """Store one FASTA record. Returns True if it duplicated an identical
    earlier record in the same file; raises on conflicting duplicates."""
    if seq_id is None:
        return False

    sequence = "".join(parts).upper()

    if seq_id in records:
        if records[seq_id] != sequence:
            raise ValueError(
                f"Conflicting sequences for ID '{seq_id}' inside file: {path}"
            )
        return True

    records[seq_id] = sequence
    return False


def read_fasta(path):
    records = {}
    current_id = None
    parts = []
    duplicates_in_file = 0

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if _commit_record(records, current_id, parts, path):
                    duplicates_in_file += 1

                current_id = normalize_id(line[1:])
                parts = []
            else:
                parts.append(line)

    if _commit_record(records, current_id, parts, path):
        duplicates_in_file += 1

    return records, duplicates_in_file


def load_sequences(folds):
    fasta_paths = sorted(
        list(TRAIN_FASTA_DIR.glob("*.fasta"))
        + list(TRAIN_FASTA_DIR.glob("*.fa"))
        + list(TRAIN_FASTA_DIR.glob("*.faa"))
    )

    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found in:\n{TRAIN_FASTA_DIR}")

    print(f"\nReading {len(fasta_paths)} FASTA file(s)...")

    sequence_map = {}
    duplicate_ids = set()
    duplicates_within_files = 0

    for fasta_path in fasta_paths:
        records, dup_in_file = read_fasta(fasta_path)
        duplicates_within_files += dup_in_file

        for seq_id, sequence in records.items():
            if not sequence:
                continue

            if seq_id in sequence_map:
                duplicate_ids.add(seq_id)

                if sequence_map[seq_id] != sequence:
                    raise ValueError(
                        f"Conflicting sequences for ID: {seq_id}\n"
                        f"File: {fasta_path}"
                    )
            else:
                sequence_map[seq_id] = sequence

    print(f"Unique FASTA IDs: {len(sequence_map):,}")
    print(
        "Duplicate FASTA IDs with identical sequence "
        f"(across files): {len(duplicate_ids):,}"
    )
    print(
        "Duplicate records with identical sequence "
        f"(within files): {duplicates_within_files:,}"
    )

    missing = set(folds["Sequence_ID"]) - set(sequence_map.keys())

    if missing:
        raise ValueError(
            f"{len(missing):,} fold IDs are missing from FASTA.\n"
            f"Examples: {sorted(missing)[:10]}"
        )

    df = folds.copy()
    df["Sequence"] = df["Sequence_ID"].map(sequence_map)

    if df["Sequence"].isna().any():
        raise ValueError("Some CV sequences could not be recovered.")

    df["Sequence"] = (
        df["Sequence"]
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )


    duplicate_sequence_mask = df["Sequence"].duplicated(keep=False)

    if duplicate_sequence_mask.any():
        duplicate_sequences = (
            df.loc[duplicate_sequence_mask]
            .groupby("Sequence")
            .agg(
                N=("Sequence_ID", "size"),
                N_Classes=("Class", "nunique"),
                N_Folds=("Fold", "nunique"),
            )
        )

        conflicting_labels = duplicate_sequences[
            duplicate_sequences["N_Classes"] > 1
        ]

        cross_fold_duplicates = duplicate_sequences[
            duplicate_sequences["N_Folds"] > 1
        ]

        print("\nExact duplicate sequence audit:")
        print(
            f"Duplicate sequence groups: {len(duplicate_sequences):,}"
        )

        if len(conflicting_labels) > 0:
            raise ValueError(
                "IDENTICAL SEQUENCES HAVE MULTIPLE CLASS LABELS.\n"
                f"Examples:\n{conflicting_labels.head(10)}"
            )

        if len(cross_fold_duplicates) > 0:
            raise ValueError(
                "IDENTICAL SEQUENCES OCCUR IN MULTIPLE OUTER FOLDS.\n"
                "This would create exact-sequence leakage.\n"
                f"Examples:\n{cross_fold_duplicates.head(10)}"
            )

        print(
            "Exact duplicate sequences are confined to a single "
            "class and a single outer fold: PASS"
        )
    else:
        print("\nExact duplicate sequence audit: NO DUPLICATE SEQUENCES FOUND")

    if df["Sequence"].str.len().eq(0).any():
        raise ValueError("Empty protein sequence detected.")


    observed = set("".join(df["Sequence"].tolist()))
    nonstandard = sorted(observed - STANDARD_AA)
    outside_vocab = sorted(observed - set(VOCAB))

    print("\nObserved sequence characters:")
    print("".join(sorted(observed)))

    if nonstandard:
        print(
            "\nNOTE: characters outside the 20 standard amino acids: "
            f"{nonstandard}"
        )

    if outside_vocab:
        print(
            "WARNING: characters outside A-Z will be mapped to the UNKNOWN "
            f"index (not dropped): {outside_vocab}"
        )

    lengths = df["Sequence"].str.len()

    print(f"\nSEQ_LENGTH = {SEQ_LENGTH}")
    print("Sequence length distribution (all classes):")
    print(
        lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1).to_string()
    )

    audit = (
        pd.DataFrame(
            {
                "Class": df["Class"].values,
                "Length": lengths.values,
                "Truncated": (lengths > SEQ_LENGTH).values,
            }
        )
        .groupby("Class")
        .agg(
            Median_Length=("Length", "median"),
            Max_Length=("Length", "max"),
            Pct_Truncated=("Truncated", lambda s: 100.0 * s.mean()),
        )
        .round(1)
    )

    print("\nPer-class length / truncation audit (post-truncation cuts the C-terminus):")
    print(audit.to_string())

    print(f"\nCV sequences recovered: {len(df):,}")

    return df

def subsample_for_smoke(df):
    """Keep up to SMOKE_PER_CLASS sequences per class. Rows are a subset of
    the original rows, so every cluster still lies in exactly one fold."""
    parts = []

    for _, group in df.groupby("Class"):
        parts.append(
            group.sample(
                n=min(len(group), SMOKE_PER_CLASS),
                random_state=RANDOM_STATE,
            )
        )

    small = pd.concat(parts).sort_index().reset_index(drop=True)

    print("\n" + "=" * 80)
    print("SMOKE TEST SUBSAMPLE")
    print("=" * 80)
    print(f"Sequences kept: {len(small):,}")
    print(small["Class"].value_counts().sort_index().to_string())

    return small

def build_model(vocab_size):
    model = Sequential(
        [
            Input(shape=(SEQ_LENGTH,)),
            Embedding(
                input_dim=vocab_size,
                output_dim=EMBED_DIM,
                mask_zero=True,
            ),
            Bidirectional(LSTM(LSTM1_UNITS, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(LSTM2_UNITS)),
            LayerNormalization(),
            Dense(
                DENSE_UNITS,
                activation="relu",
                kernel_regularizer=l2(0.001),
            ),
            Dropout(0.4),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_class_weights(y):
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_inner_split(outer_train, outer_fold, strict=True):
    splitter = StratifiedGroupKFold(
        n_splits=INNER_SPLITTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold,
    )

    splits = list(
        splitter.split(
            np.zeros(len(outer_train)),
            outer_train["Class"].values,
            outer_train["Cluster_ID"].values,
        )
    )

    inner_train_idx, inner_val_idx = splits[INNER_SPLIT_INDEX]

    inner_train = outer_train.iloc[inner_train_idx].reset_index(drop=True)
    inner_val = outer_train.iloc[inner_val_idx].reset_index(drop=True)

    overlap = set(inner_train["Cluster_ID"]) & set(inner_val["Cluster_ID"])

    if overlap:
        raise ValueError(
            "INNER CLUSTER LEAKAGE.\n"
            f"Overlapping clusters: {list(overlap)[:20]}"
        )

    for name, part in (("training", inner_train), ("validation", inner_val)):
        missing = set(CLASS_LABELS) - set(part["Class"])

        if missing:
            message = (
                f"Inner {name} set is missing class(es): {sorted(missing)}."
            )

            if strict:
                raise ValueError(message)

            print(f"WARNING (smoke test only): {message}")

    return inner_train, inner_val


class LearningRateRecorder(Callback):
    """Record the LR actually used at the START of every epoch."""

    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate

        if hasattr(lr, "numpy"):
            lr = lr.numpy()

        # round away float32 noise (0.0005 -> 0.0005000000237)
        self.learning_rates.append(float(f"{float(lr):.7g}"))


class FixedLearningRateSchedule(Callback):
    """Apply an exact epoch-by-epoch LR list (used for the refit models)."""

    def __init__(self, learning_rates):
        super().__init__()
        self.learning_rates = [float(x) for x in learning_rates]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= len(self.learning_rates):
            raise RuntimeError(
                "Training requested more epochs than the recorded LR schedule."
            )

        lr = self.learning_rates[epoch]
        optimizer_lr = self.model.optimizer.learning_rate

        if hasattr(optimizer_lr, "assign"):
            optimizer_lr.assign(lr)
        else:
            tf.keras.backend.set_value(optimizer_lr, lr)

def determine_training_schedule(
    outer_train,
    outer_fold,
    max_epochs,
    output_dir,
    smoke_test=False,
):
    print("\n" + "-" * 80)
    print("INNER TRAINING / SCHEDULE SELECTION")
    print("-" * 80)

    inner_train, inner_val = get_inner_split(
        outer_train,
        outer_fold,
        strict=not smoke_test,
    )

    print(f"Inner training   : {len(inner_train):,}")
    print(f"Inner validation : {len(inner_val):,}")

    print("\nInner training class counts:")
    print(inner_train["Class"].value_counts().sort_index().to_string())

    print("\nInner validation class counts:")
    print(inner_val["Class"].value_counts().sort_index().to_string())

    X_train = encode_sequences(inner_train["Sequence"].tolist())
    X_val = encode_sequences(inner_val["Sequence"].tolist())

    y_train = inner_train["Class"].to_numpy(dtype=np.int64)
    y_val = inner_val["Class"].to_numpy(dtype=np.int64)

    # Class weights from inner-TRAINING labels only.
    class_weights = get_class_weights(y_train)

    if WEIGHTED_VAL_LOSS:
        w_val = np.array(
            [class_weights.get(int(c), 1.0) for c in y_val],
            dtype=np.float32,
        )
        validation_data = (X_val, y_val, w_val)
    else:
        validation_data = (X_val, y_val)

    if smoke_test:
        epochs_to_run = SMOKE_EPOCHS
        early_patience = 2
        lr_patience = 1
    else:
        epochs_to_run = max_epochs
        early_patience = EARLY_STOP_PATIENCE
        lr_patience = LR_PATIENCE

    tf.keras.backend.clear_session()
    set_seed(RANDOM_STATE + outer_fold)

    model = build_model(VOCAB_SIZE)

    lr_recorder = LearningRateRecorder()

    callbacks = [
        lr_recorder,
        EarlyStopping(
            monitor="val_loss",
            patience=early_patience,
            restore_best_weights=False,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=lr_patience,
            min_lr=MIN_LR,
            verbose=1,
        ),
        TerminateOnNaN(),
    ]

    print(
        "\nTraining inner model "
        f"(weighted validation loss: {WEIGHTED_VAL_LOSS})..."
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        batch_size=BATCH_SIZE,
        epochs=epochs_to_run,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_losses = np.asarray(history.history["val_loss"], dtype=float)

    if not np.isfinite(val_losses).all():
        raise RuntimeError(
            "Non-finite val_loss in the inner run; training diverged. "
            "Epoch selection would be meaningless."
        )

    best_epoch = int(np.argmin(val_losses) + 1)
    best_val_loss = float(np.min(val_losses))

    completed_epochs = len(history.history["loss"])
    learning_rates = [float(x) for x in lr_recorder.learning_rates]

    if len(learning_rates) != completed_epochs:
        raise RuntimeError(
            "Learning-rate recorder length does not match completed epochs "
            f"({len(learning_rates)} vs {completed_epochs})."
        )

    selected_lr_schedule = learning_rates[:best_epoch]

    best_train_accuracy = float(np.max(history.history["accuracy"]))
    final_train_accuracy = float(history.history["accuracy"][-1])
    final_train_loss = float(history.history["loss"][-1])

    print("\nINNER TRAINING SUMMARY")
    print(f"Epochs completed    : {completed_epochs}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best val loss       : {best_val_loss:.6f}")
    print(f"Best train accuracy : {best_train_accuracy:.6f}")
    print(f"Final train accuracy: {final_train_accuracy:.6f}")
    print(f"Final train loss    : {final_train_loss:.6f}")
    print(f"LR schedule selected: {summarize_schedule(selected_lr_schedule)}")

    schedule = {
        "outer_fold": int(outer_fold),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "weighted_val_loss": bool(WEIGHTED_VAL_LOSS),
        "learning_rates": [float(x) for x in selected_lr_schedule],
        "inner_train_n": int(len(inner_train)),
        "inner_validation_n": int(len(inner_val)),
        "best_train_accuracy": float(best_train_accuracy),
        "final_train_accuracy": float(final_train_accuracy),
        "final_train_loss": float(final_train_loss),
        "class_weights": class_weights,
    }

    fold_dir = output_dir / f"outer_fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with open(fold_dir / "inner_training_schedule.json", "w") as handle:
        json.dump(schedule, handle, indent=4)

    with open(fold_dir / "inner_training_history.json", "w") as handle:
        json.dump(clean_history(history.history), handle, indent=2)

    del model, X_train, X_val, y_train, y_val
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    return best_epoch, selected_lr_schedule, schedule


def calculate_metrics(y_true, y_pred, y_prob):
    result = {}

    result["Accuracy"] = accuracy_score(y_true, y_pred)
    result["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)

    result["Macro_Precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    result["Macro_Recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    result["Macro_F1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    result["Weighted_F1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    try:
        result["ROC_AUC_OVR_Macro"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=CLASS_LABELS,
        )
    except ValueError as error:
        print(f"WARNING: ROC-AUC could not be computed: {error}")
        result["ROC_AUC_OVR_Macro"] = np.nan

    try:
        y_bin = np.eye(NUM_CLASSES, dtype=int)[y_true]
        result["Average_Precision_Macro"] = average_precision_score(
            y_bin, y_prob, average="macro"
        )
    except ValueError as error:
        print(f"WARNING: Average precision could not be computed: {error}")
        result["Average_Precision_Macro"] = np.nan

    return result


def run_outer_fold(df, outer_fold, max_epochs, output_dir, smoke_test=False):
    fold_start = time.time()

    tf.keras.backend.clear_session()
    gc.collect()

    fold_dir = output_dir / f"outer_fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    outer_test = df[df["Fold"] == outer_fold].reset_index(drop=True)
    outer_train = df[df["Fold"] != outer_fold].reset_index(drop=True)

    train_clusters = set(outer_train["Cluster_ID"])
    test_clusters = set(outer_test["Cluster_ID"])

    if train_clusters & test_clusters:
        raise ValueError("OUTER CLUSTER LEAKAGE.")

    print("\n" + "=" * 90)
    print(f"OUTER FOLD {outer_fold}/{N_OUTER_FOLDS}")
    print("=" * 90)
    print(f"Outer training       : {len(outer_train):,}")
    print(f"Outer validation     : {len(outer_test):,}")
    print(f"Training clusters    : {len(train_clusters):,}")
    print(f"Validation clusters  : {len(test_clusters):,}")
    print("Outer cluster separation: PASS")

    print("\nOuter-training class counts:")
    print(outer_train["Class"].value_counts().sort_index().to_string())

    print("\nOuter-validation class counts:")
    print(outer_test["Class"].value_counts().sort_index().to_string())


    best_epoch, lr_schedule, _ = determine_training_schedule(
        outer_train,
        outer_fold,
        max_epochs,
        output_dir,
        smoke_test,
    )


    print("\n" + "-" * 80)
    print("FINAL OUTER-TRAINING REFIT")
    print("-" * 80)

    X_outer_train = encode_sequences(outer_train["Sequence"].tolist())
    X_outer_test = encode_sequences(outer_test["Sequence"].tolist())

    y_outer_train = outer_train["Class"].to_numpy(dtype=np.int64)
    y_outer_test = outer_test["Class"].to_numpy(dtype=np.int64)

    final_class_weights = get_class_weights(y_outer_train)

    print("\nOuter-training class weights:")
    print(json.dumps(final_class_weights, indent=4))

    set_seed(RANDOM_STATE + outer_fold)
    model = build_model(VOCAB_SIZE)

    callbacks = [
        FixedLearningRateSchedule(lr_schedule),
        TerminateOnNaN(),
    ]

    print(f"\nTraining sequences : {len(outer_train):,}")
    print(f"Training epochs    : {best_epoch}")
    print(f"LR schedule        : {summarize_schedule(lr_schedule)}")

    # NO validation_data here: the outer fold stays untouched.
    final_history = model.fit(
        X_outer_train,
        y_outer_train,
        batch_size=BATCH_SIZE,
        epochs=best_epoch,
        class_weight=final_class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    final_train_loss = float(final_history.history["loss"][-1])
    final_train_accuracy = float(final_history.history["accuracy"][-1])

    print("\nFinal outer-training sanity check:")
    print(f"Final training loss     : {final_train_loss:.6f}")
    print(f"Final training accuracy : {final_train_accuracy:.6f}")


    print("\n" + "-" * 80)
    print("EVALUATING UNTOUCHED OUTER VALIDATION")
    print("-" * 80)

    y_prob = model.predict(X_outer_test, batch_size=BATCH_SIZE, verbose=1)

    if not np.isfinite(y_prob).all():
        raise RuntimeError("Non-finite prediction probabilities detected.")

    y_pred = np.argmax(y_prob, axis=1)

    unique_predictions = np.unique(y_pred)

    print("\nPredicted classes:")
    print(unique_predictions)

    if len(unique_predictions) == 1:
        print("\nWARNING: OUTER PREDICTIONS HAVE COLLAPSED TO A SINGLE CLASS.")

    metrics = calculate_metrics(y_outer_test, y_pred, y_prob)

    metrics.update(
        {
            "Dataset": "multiclass",
            "Outer_Fold": int(outer_fold),
            "Selected_Epoch": int(best_epoch),
            "Train_N": int(len(outer_train)),
            "Outer_Validation_N": int(len(outer_test)),
            "Train_Clusters": int(len(train_clusters)),
            "Outer_Validation_Clusters": int(len(test_clusters)),
            "Vocab_Size": int(VOCAB_SIZE),
            "Final_Train_Loss": final_train_loss,
            "Final_Train_Accuracy": final_train_accuracy,
            "Unique_Predicted_Classes": int(len(unique_predictions)),
            "Runtime_Seconds": float(time.time() - fold_start),
        }
    )

    print("\n" + "=" * 80)
    print(f"OUTER FOLD {outer_fold} RESULTS")
    print("=" * 80)

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key:32s}: {value:.6f}")
        else:
            print(f"{key:32s}: {value}")


    model.save(fold_dir / "cv_fold_model.keras")

    with open(fold_dir / "final_class_weights.json", "w") as handle:
        json.dump(final_class_weights, handle, indent=4)

    with open(fold_dir / "final_training_history.json", "w") as handle:
        json.dump(clean_history(final_history.history), handle, indent=2)

    fold_metadata = {
        "outer_fold": int(outer_fold),
        "outer_training_n": int(len(outer_train)),
        "outer_validation_n": int(len(outer_test)),
        "outer_training_clusters": int(len(train_clusters)),
        "outer_validation_clusters": int(len(test_clusters)),
        "selected_epoch": int(best_epoch),
        "learning_rate_schedule": [float(x) for x in lr_schedule],
        "sequence_length": int(SEQ_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "vocab_size": int(VOCAB_SIZE),
        "initial_learning_rate": float(LEARNING_RATE),
        "early_stopping_patience": int(EARLY_STOP_PATIENCE),
        "lr_patience": int(LR_PATIENCE),
        "lr_factor": float(LR_FACTOR),
        "minimum_lr": float(MIN_LR),
        "weighted_val_loss": bool(WEIGHTED_VAL_LOSS),
        "final_training_loss": float(final_train_loss),
        "final_training_accuracy": float(final_train_accuracy),
        "outer_validation_untouched": True,
        "independent_test_used": False,
        "full_outer_training_refit": True,
        "smoke_test": bool(smoke_test),
    }

    with open(fold_dir / "fold_metadata.json", "w") as handle:
        json.dump(fold_metadata, handle, indent=4)

    cm = confusion_matrix(y_outer_test, y_pred, labels=CLASS_LABELS)

    pd.DataFrame(
        cm,
        index=[f"Actual_{CLASS_NAMES[x]}" for x in CLASS_LABELS],
        columns=[f"Predicted_{CLASS_NAMES[x]}" for x in CLASS_LABELS],
    ).to_csv(fold_dir / "confusion_matrix.csv")

    report = classification_report(
        y_outer_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=[CLASS_NAMES[x] for x in CLASS_LABELS],
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(fold_dir / "classification_report.csv")

    outer_train[["Sequence_ID", "Class", "Fold", "Cluster_ID"]].to_csv(
        fold_dir / "outer_training_ids.csv", index=False
    )
    outer_test[["Sequence_ID", "Class", "Fold", "Cluster_ID"]].to_csv(
        fold_dir / "outer_validation_ids.csv", index=False
    )

    predictions = pd.DataFrame(
        {
            "Sequence_ID": outer_test["Sequence_ID"].values,
            "Outer_Fold": int(outer_fold),
            "True_Class": y_outer_test,
            "True_Class_Name": [CLASS_NAMES[x] for x in y_outer_test],
            "Predicted_Class": y_pred,
            "Predicted_Class_Name": [CLASS_NAMES[x] for x in y_pred],
        }
    )

    for class_id in CLASS_LABELS:
        predictions[f"Probability_{CLASS_NAMES[class_id]}"] = y_prob[:, class_id]

    predictions.to_csv(
        fold_dir / "outer_validation_predictions.csv", index=False
    )

    with open(fold_dir / "fold_metrics.json", "w") as handle:
        json.dump({k: jsonable(v) for k, v in metrics.items()}, handle, indent=4)

    del model, X_outer_train, X_outer_test, y_prob, y_pred
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics

def aggregate_fold_outputs(output_dir, expected_n, df):
    rows = []
    prediction_frames = []
    fold_schedules = []
    missing_folds = []

    for k in range(1, N_OUTER_FOLDS + 1):
        fold_dir = output_dir / f"outer_fold_{k}"

        metrics_path = fold_dir / "fold_metrics.json"
        predictions_path = fold_dir / "outer_validation_predictions.csv"
        schedule_path = fold_dir / "inner_training_schedule.json"

        if not all(p.exists() for p in (metrics_path, predictions_path, schedule_path)):
            missing_folds.append(k)
            continue

        with open(metrics_path) as handle:
            rows.append(json.load(handle))

        prediction_frames.append(
            pd.read_csv(predictions_path, dtype={"Sequence_ID": str})
        )

        with open(schedule_path) as handle:
            fold_schedules.append(json.load(handle)["learning_rates"])

    if missing_folds:
        raise FileNotFoundError(
            f"Outer fold(s) {missing_folds} are not finished in:\n{output_dir}\n"
            "Run them first (python script.py --fold K), then aggregate."
        )

    results_df = pd.DataFrame(rows)

    order = np.argsort(results_df["Outer_Fold"].values)
    results_df = results_df.iloc[order].reset_index(drop=True)
    fold_schedules = [fold_schedules[i] for i in order]

    pooled = pd.concat(prediction_frames, ignore_index=True)


    if pooled["Sequence_ID"].duplicated().any():
        raise ValueError("A sequence received more than one outer OOF prediction.")

    if len(pooled) != expected_n:
        raise ValueError(
            f"Pooled OOF prediction count {len(pooled):,} != expected {expected_n:,}."
        )

    if set(pooled["Sequence_ID"]) != set(df["Sequence_ID"]):
        raise ValueError("Pooled OOF IDs do not match the dataset IDs.")

    truth = df.set_index("Sequence_ID")["Class"]

    if not (pooled["True_Class"].values == truth.loc[pooled["Sequence_ID"]].values).all():
        raise ValueError("True classes in OOF predictions do not match the dataset.")

    return results_df, pooled, fold_schedules


def summarize_cv(results_df, pooled, output_dir, fold_schedules):
    results_df.to_csv(output_dir / "outer_fold_results.csv", index=False)

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
        "ROC_AUC_OVR_Macro",
        "Average_Precision_Macro",
        "Selected_Epoch",
        "Final_Train_Loss",
        "Final_Train_Accuracy",
    ]

    summary_rows = []

    for metric in metric_columns:
        values = pd.to_numeric(results_df[metric], errors="coerce").dropna()

        summary_rows.append(
            {
                "Metric": metric,
                "Mean": values.mean(),
                "SD": values.std(ddof=1),
                "Min": values.min(),
                "Max": values.max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "CV_mean_SD_summary.csv", index=False)

    pooled.to_csv(output_dir / "pooled_out_of_fold_predictions.csv", index=False)

    pooled_true = pooled["True_Class"].to_numpy()
    pooled_pred = pooled["Predicted_Class"].to_numpy()
    pooled_prob = pooled[
        [f"Probability_{CLASS_NAMES[x]}" for x in CLASS_LABELS]
    ].to_numpy()

    pooled_metrics = calculate_metrics(pooled_true, pooled_pred, pooled_prob)

    with open(output_dir / "pooled_out_of_fold_metrics.json", "w") as handle:
        json.dump({k: jsonable(v) for k, v in pooled_metrics.items()}, handle, indent=4)

    pooled_cm = confusion_matrix(pooled_true, pooled_pred, labels=CLASS_LABELS)

    pd.DataFrame(
        pooled_cm,
        index=[f"Actual_{CLASS_NAMES[x]}" for x in CLASS_LABELS],
        columns=[f"Predicted_{CLASS_NAMES[x]}" for x in CLASS_LABELS],
    ).to_csv(output_dir / "pooled_out_of_fold_confusion_matrix.csv")

    print("\n" + "=" * 90)
    print("CLUSTER-AWARE Bi-LSTM CV RESULTS")
    print("=" * 90)

    print("\nFold-wise results:")
    print(
        results_df[
            [
                "Outer_Fold",
                "Accuracy",
                "Balanced_Accuracy",
                "Macro_F1",
                "Weighted_F1",
                "ROC_AUC_OVR_Macro",
                "Average_Precision_Macro",
                "Selected_Epoch",
            ]
        ].to_string(index=False)
    )

    print("\nMean +/- SD (PRIMARY result):")
    for row in summary_rows:
        print(f"{row['Metric']:30s}: {row['Mean']:.6f} +/- {row['SD']:.6f}")

    print("\nPooled out-of-fold results (secondary):")
    for key, value in pooled_metrics.items():
        print(f"{key:30s}: {value:.6f}")

    epochs = results_df["Selected_Epoch"].astype(int).tolist()

    if min(epochs) > 0 and max(epochs) / min(epochs) > 2:
        print(
            "\nWARNING: Selected_Epoch varies more than 2-fold across folds "
            f"({epochs}). The epoch estimate is unstable."
        )

    for k, sched in enumerate(fold_schedules, start=1):
        if any(lr < LEARNING_RATE for lr in sched):
            print(f"NOTE: fold {k} LR was reduced before its best epoch.")

    return pooled_metrics


def train_final_model(df, results_df, fold_schedules, output_dir):
    start = time.time()

    epochs_per_fold = results_df["Selected_Epoch"].astype(int).tolist()
    final_epochs = max(1, int(round(float(np.median(epochs_per_fold)))))

    final_lr_schedule = build_median_lr_schedule(fold_schedules, final_epochs)

    print("\n" + "=" * 90)
    print("FINAL MODEL ON ALL SEQUENCES")
    print("=" * 90)
    print(f"Selected epochs per fold : {epochs_per_fold}")
    print(f"Final model epochs       : {final_epochs} (median)")
    print(f"Final LR schedule        : {summarize_schedule(final_lr_schedule)}")
    print(f"Training sequences       : {len(df):,}")

    X = encode_sequences(df["Sequence"].tolist())
    y = df["Class"].to_numpy(dtype=np.int64)

    class_weights = get_class_weights(y)

    tf.keras.backend.clear_session()
    set_seed(RANDOM_STATE)

    model = build_model(VOCAB_SIZE)

    history = model.fit(
        X,
        y,
        batch_size=BATCH_SIZE,
        epochs=final_epochs,
        class_weight=class_weights,
        callbacks=[FixedLearningRateSchedule(final_lr_schedule), TerminateOnNaN()],
        verbose=1,
    )

    final_dir = output_dir / "FINAL_MODEL"
    final_dir.mkdir(parents=True, exist_ok=True)

    model_path = final_dir / "final_model.keras"
    tokenizer_path = final_dir / "final_tokenizer.pkl"

    model.save(model_path)

    encoder = build_encoder()

    with open(tokenizer_path, "wb") as handle:
        pickle.dump(encoder, handle)

    with open(final_dir / "final_vocab.json", "w") as handle:
        json.dump(encoder, handle, indent=2)

    with open(final_dir / "final_class_weights.json", "w") as handle:
        json.dump(class_weights, handle, indent=4)

    with open(final_dir / "final_training_history.json", "w") as handle:
        json.dump(clean_history(history.history), handle, indent=2)


    with open(tokenizer_path, "rb") as handle:
        reloaded_encoder = pickle.load(handle)

    probe = encode_sequences(df["Sequence"].tolist()[:64], reloaded_encoder)

    reloaded_model = tf.keras.models.load_model(model_path)

    p_memory = model.predict(probe, batch_size=BATCH_SIZE, verbose=0)
    p_disk = reloaded_model.predict(probe, batch_size=BATCH_SIZE, verbose=0)

    max_diff = float(np.max(np.abs(p_memory - p_disk)))

    if max_diff > 1e-4:
        raise RuntimeError(
            "Reloaded final model does not reproduce predictions "
            f"(max abs diff {max_diff:.3g})."
        )

    print(f"\nSave/reload check passed (max abs diff {max_diff:.2e}).")

    metadata = {
        "training_sequences": int(len(df)),
        "epochs": int(final_epochs),
        "epochs_per_fold": epochs_per_fold,
        "learning_rate_schedule": final_lr_schedule,
        "sequence_length": int(SEQ_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "vocab_size": int(VOCAB_SIZE),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "random_state": int(RANDOM_STATE),
        "independent_test_used": False,
        "runtime_seconds": float(time.time() - start),
        "usage": (
            "Encode with encode_sequences(seqs, encoder=pickle.load("
            "open('final_tokenizer.pkl','rb'))) and predict with "
            "tf.keras.models.load_model('final_model.keras')."
        ),
    }

    with open(final_dir / "final_model_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=4)

    print("\nFinal deliverables:")
    print(f"  {model_path}")
    print(f"  {tokenizer_path}")

    del model, reloaded_model, X, y
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()


def main():
    args = parse_args()

    # Critical configuration sanity checks before any training begins.
    if NUM_CLASSES != len(CLASS_LABELS):
        raise ValueError("NUM_CLASSES and CLASS_LABELS are inconsistent.")
    if sum(EXPECTED_CLASS_COUNTS.values()) != EXPECTED_N:
        raise ValueError("EXPECTED_CLASS_COUNTS do not sum to EXPECTED_N.")
    if VOCAB_SIZE != len(VOCAB) + 2:
        raise ValueError("VOCAB_SIZE is inconsistent with VOCAB definition.")

    output_dir = OUTPUT_DIR / SMOKE_DIR_NAME if args.smoke_test else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print("\n" + "=" * 80)
        print("SMOKE TEST MODE")
        print(
            f"~{SMOKE_PER_CLASS} sequences/class, {SMOKE_EPOCHS} inner epochs, "
            "full pipeline incl. refit, aggregation and final model."
        )
        print(f"Output: {output_dir}")
        print("NO PERFORMANCE RESULTS SHOULD BE REPORTED.")
        print("=" * 80 + "\n")

    start_time = time.time()

    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    folds = load_fold_assignments()
    df = load_sequences(folds)

    if len(df) != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N:,} sequences but recovered {len(df):,}."
        )

    if args.smoke_test:
        df = subsample_for_smoke(df)

    expected_n = len(df)

    with open(output_dir / "vocab.json", "w") as handle:
        json.dump(build_encoder(), handle, indent=2)


    if not args.aggregate_only:
        folds_to_run = (
            [args.fold] if args.fold is not None
            else list(range(1, N_OUTER_FOLDS + 1))
        )

        print(f"\nFolds to run: {folds_to_run}")

        for outer_fold in folds_to_run:
            run_outer_fold(
                df,
                outer_fold,
                args.max_epochs,
                output_dir,
                args.smoke_test,
            )

        if args.fold is not None:
            print(
                f"\nFold {args.fold} finished and saved in "
                f"{output_dir / f'outer_fold_{args.fold}'}."
            )
            print(
                "When all five folds are done, run:\n"
                "    python bilstm_cluster_cv_final.py --aggregate_only"
            )
            return


    results_df, pooled, fold_schedules = aggregate_fold_outputs(
        output_dir, expected_n, df
    )

    pooled_metrics = summarize_cv(results_df, pooled, output_dir, fold_schedules)


    if not args.skip_final:
        train_final_model(df, results_df, fold_schedules, output_dir)

    runtime = time.time() - start_time
    fold_runtime = float(results_df["Runtime_Seconds"].sum())

    run_metadata = {
        "design": (
            "5-fold cluster-aware outer CV; one deterministic cluster-aware "
            "inner validation split per outer fold; inner EarlyStopping and "
            "ReduceLROnPlateau (class-weighted val loss) determine epochs and "
            "LR trajectory; new model refit on ALL outer-training sequences "
            "with the inner-derived LR schedule; untouched outer fold "
            "evaluated once; one final model trained on all sequences."
        ),
        "smoke_test": bool(args.smoke_test),
        "number_of_sequences": int(len(df)),
        "number_of_clusters": int(df["Cluster_ID"].nunique()),
        "outer_folds": int(N_OUTER_FOLDS),
        "inner_splitter_folds": int(INNER_SPLITTER_FOLDS),
        "inner_split_index": int(INNER_SPLIT_INDEX),
        "sequence_length": int(SEQ_LENGTH),
        "batch_size": int(BATCH_SIZE),
        "maximum_inner_epochs": int(args.max_epochs),
        "initial_learning_rate": float(LEARNING_RATE),
        "early_stopping_patience": int(EARLY_STOP_PATIENCE),
        "reduce_lr_patience": int(LR_PATIENCE),
        "reduce_lr_factor": float(LR_FACTOR),
        "minimum_learning_rate": float(MIN_LR),
        "weighted_val_loss": bool(WEIGHTED_VAL_LOSS),
        "vocab_size": int(VOCAB_SIZE),
        "independent_test_used": False,
        "outer_validation_used_for_training": False,
        "full_outer_training_refit": True,
        "final_model_trained": bool(not args.skip_final),
        "pooled_oof_predictions": int(len(pooled)),
        "fold_runtime_seconds_sum": fold_runtime,
        "this_invocation_runtime_seconds": float(runtime),
        "versions": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    with open(output_dir / "run_metadata.json", "w") as handle:
        json.dump(run_metadata, handle, indent=4)

    print(f"\nThis invocation runtime: {runtime / 3600:.2f} hours")
    print(f"All outputs in: {output_dir}")

    if args.smoke_test:
        print("\nSMOKE TEST COMPLETED.")


if __name__ == "__main__":
    main()
