#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Authors: Kavin S. Arulselvan,
         Priyadharshini Pulavendean,
         Vaishnavi Saravanan,
         Ragothaman M. Yennamalli, SASTRA Deemed to be University
Version: 1.0
Last Modified: 2025-05-28
Description:
  Feature extraction pipeline for protein sequence analysis.
  - Supports 20+ encoding methods for protein feature extraction
  - Handles multiple sequence descriptors and merging
  - Generates comprehensive CSV outputs with metadata
  - Includes runtime metrics and error handling

Usage:
  python feature_extraction.py --file <input.fasta> [--path <data_path>] 
  [--train <train_file>] [--label <label_file>] [--order <order_type>] 
  [--userDefinedOrder <custom_order>]

Input:
  FASTA file containing protein sequences
Output:
  CSV file with merged feature descriptors and extraction metrics
"""

import argparse
import re
import os
import pandas as pd
from collections import defaultdict
from codes import *  # Assuming encoding functions are defined in 'codes'
import time  # Importing time module for timestamps

# Define output directory
output_dir = r"/home/ragothaman/preDSLpmo_v2/Feature_extraction/independent_dataset/output_files"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
print(f"Output directory set to: {output_dir}")  # Confirming output directory

# Function to save feature to DataFrame
all_features_df = pd.DataFrame()
feature_counts = defaultdict(int)  # Dictionary to track descriptor counts per feature
total_features_added = 0  # Counter for total features added


def add_feature_to_dataframe(feature, encodings):
    global all_features_df, total_features_added
    if not encodings:
        print(f"No features to add for {feature}.")
        return

    try:
        feature_df = pd.DataFrame(encodings)

        # Rename the first column to 'Sequence ID'
        feature_df.rename(columns={feature_df.columns[0]: 'Sequence ID'}, inplace=True)

        num_descriptors = len(feature_df.columns) - 1  # Exclude sequence ID column
        feature_counts[feature] = num_descriptors  # Store descriptor count for the feature

        # Naming columns
        if all_features_df.empty:
            # For the first feature, rename the first 20 descriptors
            num_rename = min(num_descriptors, 20)  # To handle cases with fewer than 20 descriptors
            for i in range(num_rename):
                feature_df.rename(columns={feature_df.columns[i + 1]: f"{feature}_Feature_{i + 1}"}, inplace=True)
            all_features_df = feature_df
        else:
            feature_df = feature_df.iloc[:, 1:]  # Remove the sequence identifier column
            feature_df.columns = [f"{feature}_Feature_{i+1}" for i in range(len(feature_df.columns))]
            all_features_df = pd.concat([all_features_df, feature_df], axis=1)

        total_features_added += 1  # Increment feature counter

        print(f"{feature} extracted and added to the DataFrame. Descriptors added: {num_descriptors}.")

    except Exception as e:
        print(f"Error adding {feature}: {e}")


# Function to merge the first two rows into headers
def merge_headers(df):
    if df.shape[0] < 2:
        return df  # Not enough rows to merge

    new_headers = []
    for col1, col2 in zip(df.columns, df.iloc[0]):
        if pd.notnull(col2):
            new_headers.append(f"{col1}_{col2}")
        else:
            new_headers.append(col1)

    df.columns = new_headers
    df = df.drop(df.index[0])  # Drop the first row after merging headers
    return df


def save_merged_features(output_dir, input_file_path):
    try:
        if all_features_df.empty:
            print("No features extracted. The DataFrame is empty, skipping save.")
            return

        # Merge headers before saving
        merged_df = merge_headers(all_features_df)

        # Extract filename from the input path
        input_filename = os.path.basename(input_file_path)

        # Extract the name without the extension
        name_without_extension = os.path.splitext(input_filename)[0]

        # Create the output filename
        merged_file_name = f"{name_without_extension}.csv"

        merged_file_path = os.path.join(output_dir, merged_file_name)
        print(f"Attempting to save file to: {merged_file_path}")  # Debugging save path
        merged_df.to_csv(merged_file_path, index=False)
        print(f"All features merged and saved to: {merged_file_path}")

        # Report descriptor counts and total features
        print("\n--- Feature Extraction Summary ---")
        total_descriptor_count = 0  # Initialize total descriptor count
        for feature, count in feature_counts.items():
            print(f"{feature}: {count} descriptors")
            total_descriptor_count += count  # Accumulate the descriptor count

        print(f"Total features added: {total_features_added}")
        print(f"Total descriptor count: {total_descriptor_count}")  # Display the final sum

    except Exception as e:
        print(f"Error saving merged features: {e}")


# Main function
if __name__ == '__main__':
    start_time = time.time()  # Record start time

    arg_parser = argparse.ArgumentParser(description="Generate numerical representations for protein sequences")

    arg_parser.add_argument("--file", required=False, help="Input FASTA file")
    arg_parser.add_argument("--path", dest='filePath', help="Data file path for 'PSSM', 'SSEB(C)', 'Disorder(BC)', 'ASA', and 'TA' encodings")
    arg_parser.add_argument("--train", dest='trainFile', help="Training file for 'KNNprotein' or 'KNNpeptide' encodings")
    arg_parser.add_argument("--label", dest='labelFile', help="Sample label file for 'KNNprotein' or 'KNNpeptide' encodings")
    arg_parser.add_argument("--order", dest='order',
                            choices=['alphabetically', 'polarity', 'sideChainVolume', 'userDefined'],
                            help="Output order for AAC, EAAC, CKSAAP, DPC, DDE, TPC descriptors")
    arg_parser.add_argument("--userDefinedOrder", dest='userDefinedOrder',
                            help="User-defined order for AAC, EAAC, CKSAAP, DPC, DDE, TPC descriptors")

    args = arg_parser.parse_args()

    # Prompt user for input file if not provided as argument
    if not args.file:
        args.file = input("Enter the path to the input FASTA file: ")

    # Assuming readFasta is a function to read your FASTA file
    try:
        fastas = readFasta.readFasta(args.file)
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        exit()

    # Define available feature extraction methods
    feature_types = [
        'AAC', 'CKSAAP', 'DPC', 'DDE', 'TPC',
        'GAAC', 'CKSAAGP', 'GDPC', 'GTPC',
        'NMBroto', 'Moran', 'Geary',
        'CTDC', 'CTDT', 'CTDD',
        'CTriad', 'KSCTriad',
        'SOCNumber', 'QSOrder',
        'PAAC', 'APAAC']

    # Define amino acid order based on user input
    userDefinedOrder = args.userDefinedOrder if args.userDefinedOrder else 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'

    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }

    myOrder = myAAorder[args.order] if args.order else 'ACDEFGHIKLMNPQRSTVWY'

    kw = {'path': args.filePath, 'train': args.trainFile, 'label': args.labelFile, 'order': myOrder}

    # Iterate through each feature type and add to DataFrame
    for feature in feature_types:
        try:
            if feature in ['KNNprotein', 'KNNpeptide'] and (not args.trainFile or not args.labelFile):
                print(f"Skipping {feature} due to missing '--train' or '--label' files.")
                continue

            myFun = f"{feature}.{feature}(fastas, **kw)"
            encodings = eval(myFun)

            if encodings and isinstance(encodings, list) and len(encodings) > 0:
                add_feature_to_dataframe(feature, encodings)
            else:
                print(f"Skipping {feature} due to missing or invalid encodings.")
        except Exception as e:
            print(f"Skipping {feature} due to error: {str(e)}")

    print(f"\nAll features have been extracted and added to a single DataFrame.")

    # Save the merged DataFrame after feature extraction
    save_merged_features(output_dir, args.file)

    end_time = time.time()  # Record end time

    elapsed_time = end_time - start_time  # Calculate elapsed time in seconds

    print(f"\nProcess completed in {elapsed_time:.2f} seconds.")

