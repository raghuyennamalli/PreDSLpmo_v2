#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean, 
#          Vaishnavi Saravanan, 
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1
# Last Modified: 2024-05-06
# Description:
#   Extracts NCBI GenBank IDs (only those preceding 'ncbi') and JGI lines from a text file.
#Usage:
#   python extract_genbank_ids_and_jgi_lines.py
#Input:
#   Directory containing AA1_jgi_lines.txt to AA17_jgi_lines.txt files.
#Outputs (all prefixed with input filename):
#     - <input>_ncbi_ids.txt: Unique GenBank IDs
#     - <input>_jgi_lines.txt: Unique JGI-containing lines
#     - <input>_duplicates.txt: Duplicate entries and lines missing GenBank IDs

import os
import re
from collections import Counter, defaultdict

def extract_ncbi_and_jgi_lines(base_path):
    """
    Extracts NCBI GenBank IDs (preceding 'ncbi') and JGI lines from a text file.
    Outputs three files: unique NCBI IDs, unique JGI lines, and duplicates/missing IDs.
    
    Parameters
    ----------
    base_path : str
        The directory where the input file is located and outputs will be saved.
    """
    file_name = input("Enter the filename (with .txt extension): ").strip()
    file_path = os.path.join(base_path, file_name)

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found in {base_path}.")
        return

    input_basename = os.path.splitext(os.path.basename(file_name))[0]

    # Regex for GenBank IDs only if preceding 'ncbi'
    ncbi_pattern = (
        r"\b(?:[A-Z]{2,}[0-9]{5,}|NP_[0-9]{5,}|WP_[0-9]{5,}|XP_[0-9]{5,})\.\d{1}(?=\s+ncbi\b)"
    )
    jgi_pattern = r"\b\d+\s+jgi\b"

    extracted_ids = []
    missing_gene_lines = []
    jgi_lines = []
    jgi_line_counts = defaultdict(int)

    with open(file_path, 'r', encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            cleaned_line = line.strip()
            # Extract NCBI GenBank IDs (preceding 'ncbi')
            matches = re.findall(ncbi_pattern, line)
            if matches:
                extracted_ids.extend(matches)
            elif 'ncbi' in line.lower():
                missing_gene_lines.append(cleaned_line)

            # Extract JGI lines
            if re.search(jgi_pattern, line):
                jgi_lines.append(cleaned_line)
                jgi_line_counts[cleaned_line] += 1

    # Process NCBI IDs
    id_counts = Counter(extracted_ids)
    ncbi_output = os.path.join(base_path, f"{input_basename}_ncbi_ids.txt")
    with open(ncbi_output, 'w', encoding="utf-8") as outfile:
        for ncbi_id in id_counts.keys():
            outfile.write(f"{ncbi_id}\n")

    # Process JGI lines
    jgi_output = os.path.join(base_path, f"{input_basename}_jgi_lines.txt")
    with open(jgi_output, 'w', encoding="utf-8") as outfile:
        for line in jgi_line_counts.keys():
            outfile.write(f"{line}\n")

    # Process Duplicates and Missing IDs
    duplicates_output = os.path.join(base_path, f"{input_basename}_duplicates.txt")
    with open(duplicates_output, 'w', encoding="utf-8") as dupfile:
        if missing_gene_lines:
            dupfile.write("=== Missing Gene ID Lines ===\n")
            for line in missing_gene_lines:
                dupfile.write(f"{line}\n")
            dupfile.write("\n")
        dupfile.write("=== Duplicate NCBI IDs ===\n")
        for ncbi_id, count in id_counts.items():
            if count > 1:
                dupfile.write(f"{ncbi_id} (count: {count})\n")
        dupfile.write("\n=== Duplicate JGI Lines ===\n")
        for jgi_line, count in jgi_line_counts.items():
            if count > 1:
                dupfile.write(f"{jgi_line} (count: {count})\n")

    # Reporting
    total_genbank = len(extracted_ids)
    duplicate_genbank = sum(count - 1 for count in id_counts.values() if count > 1)
    unique_genbank = len(id_counts)
    total_jgi = len(jgi_lines)
    duplicate_jgi = sum(count - 1 for count in jgi_line_counts.values() if count > 1)
    unique_jgi = len(jgi_line_counts)

    print("Output Files:")
    print(f"  Unique NCBI IDs: {ncbi_output}")
    print(f"  Unique JGI lines: {jgi_output}")
    print(f"  Duplicates/missing: {duplicates_output}")
    print("\nStatistics:")
    print(f"  NCBI IDs: {total_genbank} total, {duplicate_genbank} duplicates, {unique_genbank} unique")
    print(f"  JGI lines: {total_jgi} total, {duplicate_jgi} duplicates, {unique_jgi} unique")
    if missing_gene_lines:
        print("\nLines with 'ncbi' but missing gene IDs:")
        for idx, line in enumerate(missing_gene_lines, 1):
            print(f"{idx}. {line}")
    else:
        print("\nNo lines with missing gene IDs.")

if __name__ == "__main__":
    # Set base path for all file operations
    BASE_PATH = r"C:\Users\HP\Desktop\major_project\Manuscript_1"
    os.chdir(BASE_PATH)
    extract_ncbi_and_jgi_lines(BASE_PATH)
