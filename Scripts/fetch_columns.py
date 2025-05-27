#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-04-14
# Description:
#   This script interactively extracts user-specified columns from a family-specific
#   combined features CSV file (e.g., AA1_combined_features.csv) and saves the result
#   as <FAMILY>_family_features.csv in the same directory.
#
# Usage:
#   python fetch_columns.py
#   (Follow the prompts to enter family name and columns to extract)
#
# Input:
#   <FAMILY>_combined_features.csv (e.g., AA1_combined_features.csv) in the specified directory.
# Output:
#   <FAMILY>_family_features.csv (e.g., AA1_family_features.csv) in the same directory.


import pandas as pd
import os

# Get base directory from user
base_dir = "path/to/your/directory"

# Get family name from user (e.g., AA1, AA2)
while True:
    family_name = input("Enter family name (e.g., AA1, AA2): ").strip().upper()
    if not family_name.startswith("AA"):
        print("Family name must start with 'AA' (e.g., AA1, AA17)")
        continue
    break

# Construct file paths
input_filename = f"{family_name}_combined_features.csv"
output_filename = f"{family_name}_family_features.csv"
input_path = os.path.join(base_dir, input_filename)
output_path = os.path.join(base_dir, output_filename)

# Check if input file exists
if not os.path.isfile(input_path):
    print(f"Error: Input file '{input_filename}' not found in {base_dir}")
    exit()

# Read only the header to get available columns
try:
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
except FileNotFoundError:
    print(f"Error: Could not read headers from {input_path}")
    exit()

# Rest of the code remains the same from here...
first_col = header[0]

# Prompt user for number of columns to fetch
while True:
    try:
        num_cols = int(input('Enter number of columns to fetch: '))
        if num_cols < 0:
            print("Please enter a positive number!")
            continue
        break
    except ValueError:
        print("Invalid input! Please enter a valid integer.")

# Prompt user for column names
columns_to_fetch = []
for i in range(num_cols):
    while True:
        col_name = input(f'Enter column {i+1} name: ').strip()
        if col_name in header:
            columns_to_fetch.append(col_name)
            break
        print(f"Column '{col_name}' not found in CSV header. Valid columns are:")
        print(', '.join(header))

# Prepare columns to read
cols_to_read = [first_col] + columns_to_fetch

# Read data with selected columns
try:
    df = pd.read_csv(input_path, usecols=cols_to_read)
except ValueError as e:
    print(f"Error: {e}")
    existing_cols = [col for col in cols_to_read if col in header]
    df = pd.read_csv(input_path, usecols=existing_cols)

# Save filtered data
df.to_csv(output_path, index=False)
print(f'Successfully saved {len(cols_to_read)-1} columns to {output_path}')
