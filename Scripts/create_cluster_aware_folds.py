#!/usr/bin/env python3

import os
import sys
import glob
import shutil
import subprocess
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


BASE_DIR = (
    "/home/ragothaman/Sweety/LPMO/"
    "New_Version/Cluster_Aware_Spliting_Changed"
)

TRAINING_DIR = os.path.join(
    BASE_DIR,
    "Training_Validation_Final",
    "Training"
)

CLEANED_FASTA_DIR = os.path.join(
    TRAINING_DIR,
    "cleaned_fasta"
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "Training_Validation_Final",
    "Feature_Extraction"
)

BINARY_EFS_DIR = os.path.join(
    FEATURE_DIR,
    "EFS",
    "binary_datasets"
)

MULTICLASS_CSV = os.path.join(
    FEATURE_DIR,
    "combined_dataset",
    "multiclass_training",
    "PredLPMO_multiclass_training_clean.csv"
)

OUTPUT_DIR = os.path.join(
    FEATURE_DIR,
    "Cluster_Aware_CV"
)

DATASET_FASTA_DIR = os.path.join(
    OUTPUT_DIR,
    "dataset_fastas"
)

CDHIT_DIR = os.path.join(
    OUTPUT_DIR,
    "cdhit"

)

FOLD_DIR = os.path.join(
    OUTPUT_DIR,
    "fold_assignments"
)


CDHIT_IDENTITY = 0.70
CDHIT_WORD_SIZE = 2
CDHIT_THREADS = 0
CDHIT_MEMORY = 0

N_SPLITS = 5
RANDOM_STATE = 42


BINARY_FAMILIES = [
    "AA9",
    "AA10",
    "AA11",
    "AA13",
    "AA14",
    "AA15",
    "AA16",
    "AA17"
]


for directory in [
    OUTPUT_DIR,
    DATASET_FASTA_DIR,
    CDHIT_DIR,
    FOLD_DIR
]:
    os.makedirs(
        directory,
        exist_ok=True
    )


def read_fasta(fasta_file):

    records = []

    header = None
    sequence_parts = []

    with open(
        fasta_file,
        "r",
        errors="ignore"
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if header is not None:

                    sequence = "".join(
                        sequence_parts
                    ).upper()

                    records.append(
                        (
                            header,
                            sequence
                        )
                    )

                header = line[1:].strip()

                sequence_parts = []

            else:

                sequence_parts.append(
                    line
                )

        if header is not None:

            sequence = "".join(
                sequence_parts
            ).upper()

            records.append(
                (
                    header,
                    sequence
                )
            )

    return records


def get_fasta_id(header):


    return header.split()[0].strip()


def build_sequence_index():

    print("\n" + "=" * 80)
    print("BUILDING CLEANED TRAINING FASTA INDEX")
    print("=" * 80)

    fasta_files = sorted(
        glob.glob(
            os.path.join(
                CLEANED_FASTA_DIR,
                "*.fasta"
            )
        )
    )

    if not fasta_files:

        print(
            "\nERROR: No cleaned FASTA files found:"
        )

        print(
            CLEANED_FASTA_DIR
        )

        sys.exit(1)

    print(
        f"\nFound {len(fasta_files)} cleaned "
        "training FASTA files."
    )

    sequence_index = {}

    duplicate_same_sequence = 0
    conflicting_sequences = {}

    total_records = 0

    for fasta_file in fasta_files:

        filename = os.path.basename(
            fasta_file
        )

        print(
            f"Reading: {filename}"
        )

        records = read_fasta(
            fasta_file
        )

        total_records += len(records)

        for header, sequence in records:

            seq_id = get_fasta_id(
                header
            )

            if seq_id not in sequence_index:

                sequence_index[seq_id] = {
                    "sequence": sequence,
                    "source_files": [fasta_file]
                }

            else:

                existing_sequence = (
                    sequence_index[seq_id][
                        "sequence"
                    ]
                )

                if existing_sequence == sequence:

                    duplicate_same_sequence += 1

                    if fasta_file not in (
                        sequence_index[seq_id][
                            "source_files"
                        ]
                    ):

                        sequence_index[
                            seq_id
                        ][
                            "source_files"
                        ].append(
                            fasta_file
                        )

                else:

                    conflicting_sequences.setdefault(
                        seq_id,
                        []
                    ).append(
                        {
                            "file": fasta_file,
                            "sequence": sequence
                        }
                    )

    print(
        f"\nTotal FASTA records read: "
        f"{total_records}"
    )

    print(
        f"Unique sequence IDs: "
        f"{len(sequence_index)}"
    )

    print(
        f"Repeated IDs with identical sequence: "
        f"{duplicate_same_sequence}"
    )

    print(
        f"IDs with conflicting sequences: "
        f"{len(conflicting_sequences)}"
    )

    if conflicting_sequences:

        print(
            "\nERROR: The following IDs have "
            "different sequences in different "
            "FASTA records:"
        )

        for seq_id, records in list(
            conflicting_sequences.items()
        )[:20]:

            print(
                f"\nSequence ID: {seq_id}"
            )

            for record in records:

                print(
                    f"  {record['file']}"
                )

        print(
            "\nThese are genuine sequence conflicts. "
            "The script will stop."
        )

        sys.exit(1)

    print(
        "\nFASTA index created successfully."
    )

    return sequence_index


def create_exact_dataset_fasta(
    dataframe,
    sequence_index,
    output_fasta,
    dataset_name
):

    print("\n" + "-" * 80)
    print(
        f"CREATING EXACT FASTA: {dataset_name}"
    )
    print("-" * 80)

    missing_ids = []
    written_ids = set()

    with open(
        output_fasta,
        "w"
    ) as output:

        for sequence_id in dataframe[
            "Sequence ID_#"
        ]:

            sequence_id = str(
                sequence_id
            ).strip()

            if sequence_id in written_ids:

                print(
                    f"\nERROR: Duplicate ID in CSV: "
                    f"{sequence_id}"
                )

                sys.exit(1)

            if sequence_id not in sequence_index:

                missing_ids.append(
                    sequence_id
                )

                continue

            sequence = sequence_index[
                sequence_id
            ]["sequence"]

            output.write(
                f">{sequence_id}\n"
            )

            for start in range(
                0,
                len(sequence),
                80
            ):

                output.write(
                    sequence[
                        start:start + 80
                    ]
                    + "\n"
                )

            written_ids.add(
                sequence_id
            )

    print(
        f"\nCSV sequences: "
        f"{len(dataframe)}"
    )

    print(
        f"FASTA sequences written: "
        f"{len(written_ids)}"
    )

    print(
        f"Missing sequences: "
        f"{len(missing_ids)}"
    )

    if missing_ids:

        print(
            "\nFirst missing IDs:"
        )

        for seq_id in missing_ids[:20]:

            print(
                f"  {seq_id}"
            )

        return False

    if len(written_ids) != len(dataframe):

        print(
            "\nERROR: Number of written sequences "
            "does not equal CSV sequence count."
        )

        return False

    print(
        "\nExact FASTA creation: PASS"
    )

    return True

def check_cd_hit():

    cd_hit = shutil.which(
        "cd-hit"
    )

    if cd_hit is None:

        print(
            "\nERROR: cd-hit is not available."
        )

        print(
            "Run:"
        )

        print(
            "which cd-hit"
        )

        sys.exit(1)

    print(
        f"\nCD-HIT: {cd_hit}"
    )

    return cd_hit


def run_cd_hit(
    cd_hit,
    input_fasta,
    output_fasta,
    dataset_name
):

    print("\n" + "-" * 80)
    print(
        f"CD-HIT 70%: {dataset_name}"
    )
    print("-" * 80)

    command = [
        cd_hit,

        "-i",
        input_fasta,

        "-o",
        output_fasta,

        "-c",
        str(CDHIT_IDENTITY),

        "-n",
        str(CDHIT_WORD_SIZE),

        "-d",
        "0",

        "-T",
        str(CDHIT_THREADS),

        "-M",
        str(CDHIT_MEMORY),

        "-g",
        "1"
    ]

    print(
        "\nRunning:"
    )

    print(
        " ".join(command)
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(
        result.stdout
    )

    if result.returncode != 0:

        print(
            "\nERROR: CD-HIT failed."
        )

        sys.exit(1)

    cluster_file = (
        output_fasta + ".clstr"
    )

    if not os.path.exists(
        cluster_file
    ):

        print(
            "\nERROR: CD-HIT .clstr file "
            "was not generated."
        )

        sys.exit(1)

    print(
        "\nCD-HIT completed successfully."
    )


def parse_clusters(
    cluster_file
):

    sequence_to_cluster = {}

    current_cluster = None

    with open(
        cluster_file,
        "r",
        errors="ignore"
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:

                continue

            if line.startswith(">Cluster"):

                current_cluster = (
                    line[1:].strip()
                )

                continue

            match = re.search(
                r">(.+?)\.\.\.",
                line
            )

            if match is None:

                continue

            sequence_id = (
                match.group(1)
                .strip()
            )

            if sequence_id in sequence_to_cluster:

                print(
                    "\nERROR: Sequence appears "
                    "in more than one CD-HIT cluster:"
                )

                print(
                    sequence_id
                )

                sys.exit(1)

            sequence_to_cluster[
                sequence_id
            ] = current_cluster

    return sequence_to_cluster


def make_folds(
    dataframe,
    cluster_map,
    dataset_name
):

    print("\n" + "-" * 80)
    print(
        f"CREATING CLUSTER-AWARE FOLDS: "
        f"{dataset_name}"
    )
    print("-" * 80)

    result = dataframe[
        [
            "Sequence ID_#",
            "Class"
        ]
    ].copy()

    result[
        "Sequence ID_#"
    ] = result[
        "Sequence ID_#"
    ].astype(str).str.strip()

    result[
        "Cluster_ID"
    ] = result[
        "Sequence ID_#"
    ].map(
        cluster_map
    )


    missing = result[
        result["Cluster_ID"].isna()
    ]

    if len(missing) > 0:

        print(
            "\nERROR: Some sequences have "
            "no cluster assignment."
        )

        print(
            missing.head(20).to_string(
                index=False
            )
        )

        return None

    print(
        f"\nSequences: "
        f"{len(result)}"
    )

    print(
        f"Clusters: "
        f"{result['Cluster_ID'].nunique()}"
    )


    clusters_per_class = (
        result
        .groupby("Class")
        ["Cluster_ID"]
        .nunique()
        .sort_index()
    )

    print(
        "\nClusters per class:"
    )

    for class_id, number in (
        clusters_per_class.items()
    ):

        sequence_count = (
            result["Class"] == class_id
        ).sum()

        print(
            f"  Class {class_id}: "
            f"{number} clusters, "
            f"{sequence_count} sequences"
        )


    insufficient = (
        clusters_per_class[
            clusters_per_class < N_SPLITS
        ]
    )

    if len(insufficient) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "The following class(es) contain "
            "fewer than 5 independent CD-HIT clusters:"
        )

        for class_id, number in (
            insufficient.items()
        ):

            print(
                f"  Class {class_id}: "
                f"{number} clusters"
            )

        print(
            "\nFive-fold cluster-aware CV cannot "
            "guarantee representation of these "
            "classes in every fold."
        )

        return None


    X_dummy = np.zeros(
        (len(result), 1)
    )

    y = result[
        "Class"
    ].values

    groups = result[
        "Cluster_ID"
    ].values

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = np.full(
        len(result),
        -1,
        dtype=int
    )

    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(
            X_dummy,
            y,
            groups
        ),
        start=1
    ):

        folds[
            test_index
        ] = fold_number

    if np.any(
        folds == -1
    ):

        print(
            "\nERROR: Some sequences "
            "were not assigned a fold."
        )

        sys.exit(1)

    result[
        "Fold"
    ] = folds


    cluster_fold_count = (
        result
        .groupby(
            "Cluster_ID"
        )["Fold"]
        .nunique()
    )

    split_clusters = (
        cluster_fold_count > 1
    ).sum()

    print(
        f"\nClusters split across folds: "
        f"{split_clusters}"
    )

    if split_clusters != 0:

        print(
            "\nFATAL ERROR:"
        )

        print(
            "At least one CD-HIT cluster "
            "was assigned to multiple folds."
        )

        sys.exit(1)

    print(
        "Cluster integrity: PASS"
    )


    summary_rows = []

    for fold in range(
        1,
        N_SPLITS + 1
    ):

        fold_data = result[
            result["Fold"] == fold
        ]

        row = {
            "Dataset":
                dataset_name,

            "Fold":
                fold,

            "Sequences":
                len(fold_data),

            "Clusters":
                fold_data[
                    "Cluster_ID"
                ].nunique()
        }

        for class_id in sorted(
            result["Class"].unique()
        ):

            row[
                f"Class_{class_id}"
            ] = (
                fold_data[
                    "Class"
                ] == class_id
            ).sum()

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        "\nFold distribution:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return result, summary


def process_dataset(
    dataframe,
    dataset_name,
    fasta_prefix,
    cd_hit,
    class_column
):

    dataframe = dataframe.copy()

    dataframe[
        "Sequence ID_#"
    ] = dataframe[
        "Sequence ID_#"
    ].astype(str).str.strip()


    if dataframe[
        "Sequence ID_#"
    ].duplicated().any():

        print(
            f"\nERROR: Duplicate Sequence IDs "
            f"in {dataset_name}"
        )

        sys.exit(1)


    exact_fasta = os.path.join(
        DATASET_FASTA_DIR,
        f"{fasta_prefix}.fasta"
    )

    if not create_exact_dataset_fasta(
        dataframe,
        SEQUENCE_INDEX,
        exact_fasta,
        dataset_name
    ):

        print(
            f"\nERROR: Exact FASTA creation "
            f"failed for {dataset_name}"
        )

        sys.exit(1)


    clustered_fasta = os.path.join(
        CDHIT_DIR,
        f"{fasta_prefix}_70pct.fasta"
    )

    run_cd_hit(
        cd_hit,
        exact_fasta,
        clustered_fasta,
        dataset_name
    )


    cluster_map = parse_clusters(
        clustered_fasta + ".clstr"
    )

    print(
        f"\nCD-HIT assignments: "
        f"{len(cluster_map)}"
    )


    dataset_ids = set(
        dataframe[
            "Sequence ID_#"
        ]
    )

    cluster_ids = set(
        cluster_map.keys()
    )

    missing_clusters = (
        dataset_ids - cluster_ids
    )

    extra_clusters = (
        cluster_ids - dataset_ids
    )

    if missing_clusters:

        print(
            "\nERROR: Dataset sequences "
            "missing from CD-HIT clusters:"
        )

        for seq_id in list(
            missing_clusters
        )[:20]:

            print(
                f"  {seq_id}"
            )

        sys.exit(1)

    if extra_clusters:

        print(
            "\nERROR: CD-HIT contains "
            "sequences not present in dataset."
        )

        sys.exit(1)

    print(
        "CD-HIT coverage: 100%"
    )


    fold_result = make_folds(
        dataframe,
        cluster_map,
        dataset_name
    )

    if fold_result is None:

        return False

    assignments, summary = (
        fold_result
    )


    assignment_file = os.path.join(
        FOLD_DIR,
        f"{fasta_prefix}_cluster_aware_5fold.csv"
    )

    summary_file = os.path.join(
        FOLD_DIR,
        f"{fasta_prefix}_fold_summary.csv"
    )

    assignments.to_csv(
        assignment_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        assignment_file
    )

    print(
        summary_file
    )

    return True

def main():

    global SEQUENCE_INDEX

    print("=" * 80)
    print("CLUSTER-AWARE 5-FOLD PREPARATION")
    print("=" * 80)

    print(
        "\nCD-HIT identity: 70%"
    )

    print(
        "CV: StratifiedGroupKFold"
    )

    print(
        "Folds: 5"
    )

    print(
        "Random state: 42"
    )


    cd_hit = check_cd_hit()


    SEQUENCE_INDEX = (
        build_sequence_index()
    )


    print("\n" + "=" * 80)
    print("MULTICLASS DATASET")
    print("=" * 80)

    multiclass_df = pd.read_csv(
        MULTICLASS_CSV
    )

    print(
        f"\nMulticlass sequences: "
        f"{len(multiclass_df)}"
    )

    if "Class" not in multiclass_df.columns:

        print(
            "\nERROR: Class column missing."
        )

        sys.exit(1)

    multiclass_ok = process_dataset(
        multiclass_df,
        "Multiclass 9-class",
        "multiclass_training",
        cd_hit,
        "Class"
    )

    if not multiclass_ok:

        print(
            "\nMulticlass fold generation "
            "was not completed."
        )


    successful = []
    failed = []

    for family in BINARY_FAMILIES:

        print("\n" + "=" * 80)
        print(
            f"{family} vs Non-LPMO"
        )
        print("=" * 80)

        csv_file = os.path.join(
            BINARY_EFS_DIR,
            f"EFS_{family}_vs_NonLPMO_training.csv"
        )

        if not os.path.exists(
            csv_file
        ):

            print(
                "\nERROR: Binary dataset not found:"
            )

            print(
                csv_file
            )

            failed.append(
                family
            )

            continue

        binary_df = pd.read_csv(
            csv_file
        )

        if "Binary_Class" not in (
            binary_df.columns
        ):

            print(
                "\nERROR: Binary_Class column "
                "missing."
            )

            failed.append(
                family
            )

            continue

        binary_df = binary_df.rename(
            columns={
                "Binary_Class": "Class"
            }
        )

        print(
            f"\nSequences: "
            f"{len(binary_df)}"
        )

        print(
            "\nClass distribution:"
        )

        print(
            binary_df[
                "Class"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        success = process_dataset(
            binary_df,
            f"{family} vs Non-LPMO",
            f"{family}_vs_NonLPMO_training",
            cd_hit,
            "Class"
        )

        if success:

            successful.append(
                family
            )

        else:

            failed.append(
                family
            )


    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        "\nSuccessful binary datasets:"
    )

    for family in successful:

        print(
            f"  {family}"
        )

    print(
        f"\nSuccessful: "
        f"{len(successful)}/8"
    )

    if failed:

        print(
            "\nNot completed:"
        )

        for family in failed:

            print(
                f"  {family}"
            )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFold assignments:"
    )

    print(
        FOLD_DIR
    )

    print("=" * 80)


if __name__ == "__main__":

    main()
