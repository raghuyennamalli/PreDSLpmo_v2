#!/usr/bin/env python3
# Authors: Vaishnavi Saravanan
# Version: 1.0
# Last Modified: 2025-04-14
# Description:
#   This script prepares multiclass training and validation datasets for machine learning.
#   It reads split information from a CSV file, fetches family feature files, assigns class labels,
#   samples sequences for training and validation, and allows interactive selection of feature columns.
#   Outputs are saved in a 'split_output' subdirectory.
#
# Usage:
#   python split_code_for_generating_train_and_val_files.py
#   (Follow the prompts to enter split file name and feature columns)
#
# Input:
#   - Split info CSV (e.g., data_split.csv) with columns: Family, NCBI_train, NCBI_val, JGI_train, JGI_val
#   - <FAMILY>_family_features.csv for each family in the specified directory
# Output:
#   - training_multiclass.csv, validation_multiclass.csv, filtered_training_multiclass.csv, filtered_validation_multiclass.csv

import os
import pandas as pd

# Paths
base_path = "path/to/your/directory"
split_base_path = "path/to/your/directory"

# Load split info file
split_file_name = input("Enter the name of the split CSV file (e.g., data_split.csv): ") 
split_file_path = os.path.join(split_base_path, split_file_name)

if not os.path.isfile(split_file_path):
    print(f"Error: {split_file_path} not found.")
    exit()

split_info_df = pd.read_csv(split_file_path)

# Family to class mapping
family_class_map = {
    'AA9': 0, 'AA10': 1, 'AA11': 2, 'AA13': 3,
    'AA14': 4, 'AA15': 5, 'AA16': 6, 'AA17': 7
}
default_class = 8

train_dfs = []
val_dfs = []
train_counts = {}
val_counts = {}

for i in range(len(split_info_df)):
    family = split_info_df.iloc[i]['Family']
    family_file = family + '_family_features.csv'
    family_class = family_class_map.get(family, default_class)

    family_path = os.path.join(base_path, family_file)
    if not os.path.isfile(family_path):
        print(f"Warning: {family_file} not found. Skipping...")
        continue

    df = pd.read_csv(family_path)

    # Drop identifier columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif df.columns[0].lower() in ['id', 'identifier', 'seq_id', 'sequence']:
        df = df.drop(columns=[df.columns[0]])

    df['class'] = family_class
    df = df[['class'] + [col for col in df.columns if col != 'class']]

    train_ncbi = int(split_info_df.iloc[i]['NCBI_train'])
    val_ncbi = int(split_info_df.iloc[i]['NCBI_val'])
    train_jgi = int(split_info_df.iloc[i]['JGI_train'])
    val_jgi = int(split_info_df.iloc[i]['JGI_val'])

    total_train = train_ncbi + train_jgi
    total_val = val_ncbi + val_jgi

    if total_train + total_val > len(df):
        print(f"Error: Not enough sequences in {family_file}. Skipping...")
        continue

    train = df.sample(n=total_train, random_state=42)
    val = df.drop(train.index).sample(n=total_val, random_state=42)

    train_dfs.append(train)
    val_dfs.append(val)

    train_counts[family] = total_train
    val_counts[family] = total_val

# Combine and shuffle
training_set = pd.concat(train_dfs, ignore_index=True).sort_values(by='class').reset_index(drop=True)
validation_set = pd.concat(val_dfs, ignore_index=True).sort_values(by='class').reset_index(drop=True)

# Save output
output_dir = os.path.join(base_path, "split_output")
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "training_multiclass.csv")
val_path = os.path.join(output_dir, "validation_multiclass.csv")
training_set.to_csv(train_path, index=False)
validation_set.to_csv(val_path, index=False)

# Summary
print("\n Multiclass splitting complete.")
print(f"Training and validation files saved in: {output_dir}")
print(f"- Training: {os.path.basename(train_path)}")
print(f"- Validation: {os.path.basename(val_path)}")

print("\n Sequence counts per family per set:")
print("Training Set:")
for fam, count in train_counts.items():
    label = family_class_map.get(fam, default_class)
    print(f"- {fam} (class {label}): {count} sequences")

print("\nValidation Set:")
for fam, count in val_counts.items():
    label = family_class_map.get(fam, default_class)
    print(f"- {fam} (class {label}): {count} sequences")
