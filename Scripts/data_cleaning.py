#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-06
# Description:
#   Cleans a FASTA file by removing sequences that contain 'X' in the sequence
#   or the word 'partial' in the description.
#   Outputs a cleaned FASTA file with '_cleaned' appended before the extension
#   in the same directory as the input file.
#
# Usage:
#   python data_cleaning.py
#   (Then enter the full path to the FASTA file when prompted)
#
# Input:
#   A FASTA file.
# Output:
#   <input_basename>_cleaned<extension> in the same directory as the input.

import os
import re
from Bio import SeqIO

def clean_fasta_unified():
    """
    Cleans a FASTA file by removing sequences containing 'X' in the sequence
    or 'partial' in the description. Outputs a cleaned file with '_cleaned'
    appended before the extension in the same directory as the input file.
    """
    file_path = input("Enter the full path to your FASTA file: ").strip()

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    dir_name = os.path.dirname(file_path)
    base_name, file_ext = os.path.splitext(os.path.basename(file_path))

    partial_pattern = re.compile(r"partial", re.IGNORECASE)
    x_pattern = re.compile(r"X")

    cleaned_sequences = []
    for seq_record in SeqIO.parse(file_path, "fasta"):
        sequence_str = str(seq_record.seq)
        description = seq_record.description
        if not x_pattern.search(sequence_str) and not partial_pattern.search(description):
            cleaned_sequences.append(seq_record)

    output_file = os.path.join(dir_name, f"{base_name}_cleaned{file_ext}")

    with open(output_file, "w") as out_f:
        SeqIO.write(cleaned_sequences, out_f, "fasta")

    original_count = sum(1 for _ in SeqIO.parse(file_path, "fasta"))
    retained_count = len(cleaned_sequences)

    print(f"Cleaned file created: {output_file}")
    print(f"Original sequences: {original_count}")
    print(f"Retained sequences: {retained_count}")

if __name__ == "__main__":
    clean_fasta_unified()
