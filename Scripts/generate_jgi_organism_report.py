#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean, 
#          Vaishnavi Saravanan, 
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-06
# Description:
#   Processes JGI organism data from multiple tab-delimited text files and generates a summary CSV report.
#   The script counts the number of JGI IDs associated with each organism in each input file.
#
# Usage:
#   python generate_jgi_organism_report.py
#   (You will be prompted for the input directory and output directory.)
#
# Input:
#   Directory containing AA1_jgi_lines.txt to AA17_jgi_lines.txt files.
# Output:
#   organism_report.csv in the specified output directory.

import os
import csv
from collections import defaultdict

def process_organisms(base_dir, output_dir):
    """
    Processes JGI organism data files and generates a CSV report.

    Parameters
    ----------
    base_dir : str
        Directory containing the input AA*.txt files.
    output_dir : str
        Directory where the output CSV will be saved.

    Returns
    -------
    str
        Path to the generated organism_report.csv file.
    """
    input_files = [f"AA{i}_jgi_lines.txt" for i in range(1, 18)]  # AA1-AA17
    organism_counts = defaultdict(lambda: defaultdict(int))

    for file_name in input_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.isfile(file_path):
            print(f"Warning: Missing file {file_name}. Skipping.")
            continue

        with open(file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 5:
                    organism = parts[2].strip()
                    jgi_id = parts[3].strip()
                    if jgi_id.isdigit():
                        organism_counts[organism][file_name] += 1
                    else:
                        print(f"Warning: Invalid JGI ID '{jgi_id}' in {file_name} line {line_num}. Skipping.")

    csv_path = os.path.join(output_dir, "organism_report.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = ["Organism"] + input_files
        writer.writerow(headers)
        for organism, counts in organism_counts.items():
            row = [organism] + [counts.get(fname, 0) for fname in input_files]
            writer.writerow(row)

    print(f"Organism report generated with {len(organism_counts)} unique organisms.")
    print(f"CSV report saved at: {csv_path}")
    return csv_path

def main():
    """
    Main function to execute the organism report generation.
    Prompts the user for input and output directories.
    """
    print("=" * 60)
    print("JGI Organism Report Generator")
    print("=" * 60)

    base_dir = input("Enter the input directory path: ").strip()
    output_dir = input("Enter the output directory path: ").strip()

    if not os.path.isdir(base_dir):
        print(f"Error: Input directory '{base_dir}' does not exist.")
        return
    os.makedirs(output_dir, exist_ok=True)

    process_organisms(base_dir, output_dir)

    print("Processing complete.")

if __name__ == "__main__":
    main()
