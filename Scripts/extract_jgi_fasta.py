#!/usr/bin/env python3
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean, 
#          Vaishnavi Saravanan, 
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-05-06
# Description:
#   Extracts organism-specific sequences from multiple metadata and FASTA files.
#   All paths and filenames are interactively provided by the user.
#
# Usage:
#   python extract_jgi_fasta.py
#   (Follow the prompts for all required inputs.)

import os
import re
from typing import Set

def process_single_file(input_file: str, fasta_file: str, output_dir: str, target_org: str) -> int:
    """
    Processes a single metadata file to extract organism-specific sequences.
    """
    organism_ids: Set[str] = set()
    target_org_cleaned = target_org.strip().lower()
    found_organism = False

    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                current_org = parts[2].strip().lower()
                if target_org_cleaned in current_org:
                    jgi_ids = re.findall(r'\d+', parts[3].strip())
                    organism_ids.update(jgi_ids)
                    found_organism = True

        if not found_organism:
            print(f"Organism '{target_org}' not found in {input_file}")
            return 0

    count = 0
    current_header = ""
    current_seq = []
    existing_files = set(os.listdir(output_dir))
    sequence_counts = {}

    with open(fasta_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                if current_header:
                    count = _write_sequence(current_header, ''.join(current_seq),
                                           organism_ids, output_dir, existing_files,
                                           sequence_counts, count)
                current_header = line.strip()
                current_seq = []
            else:
                current_seq.append(line.strip())

        if current_header:
            count = _write_sequence(current_header, ''.join(current_seq),
                                   organism_ids, output_dir, existing_files,
                                   sequence_counts, count)
    return count

def _write_sequence(header: str, sequence: str, organism_ids: Set[str],
                    output_dir: str, existing_files: Set[str],
                    sequence_counts: dict, count: int) -> int:
    header_id = re.search(r'\|(\d+)\|', header)
    if header_id and header_id.group(1) in organism_ids:
        jgi_id = header_id.group(1)
        sequence_counts[jgi_id] = sequence_counts.get(jgi_id, 0) + 1
        occurrence = sequence_counts[jgi_id]

        filename = f"{jgi_id}.fasta"
        if occurrence > 1:
            filename = f"{jgi_id}_{occurrence}.fasta"
        while filename in existing_files:
            occurrence += 1
            filename = f"{jgi_id}_{occurrence}.fasta"
        existing_files.add(filename)
        fasta_path = os.path.join(output_dir, filename)
        with open(fasta_path, 'w') as outfile:
            outfile.write(f"{header}\n{sequence}\n")
        print(f"Extracted sequence: {filename}")
        return count + 1
    return count

def prompt_path(prompt_text, must_exist=True, is_file=False):
    while True:
        path = input(prompt_text).strip()
        if must_exist:
            if is_file and not os.path.isfile(path):
                print(f"File not found: {path}")
            elif not is_file and not os.path.isdir(path):
                print(f"Directory not found: {path}")
            else:
                return path
        else:
            return path

def main():
    print("=== JGI FASTA Sequence Extractor ===")
    input_dir = prompt_path("Enter the directory containing AA*_jgi.txt files: ")
    output_dir = prompt_path("Enter the directory where output folders should be created: ", must_exist=False)
    fasta_dir = prompt_path("Enter the directory containing Organism FASTA files: ")
    organism = input("Enter the target organism name: ").strip()
    fasta_file_name = input("Enter the Organism FASTA filename (e.g., sequences.fasta): ").strip()
    fasta_file = os.path.join(fasta_dir, fasta_file_name)
    if not os.path.isfile(fasta_file):
        print(f"Error: FASTA file not found at {fasta_file}")
        return

    input_files = [f"AA{n}_jgi_lines.txt" for n in range(1, 18)]
    log_file = os.path.join(output_dir, "processing.log")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(log_file, 'a') as log:
        log.write(f"\nProcessing initiated for {organism}\n")
        for input_file in input_files:
            input_path = os.path.join(input_dir, input_file)
            aa_folder = f"{input_file.split('_')[0]}_jgi"
            aa_output_dir = os.path.join(output_dir, aa_folder)
            if not os.path.exists(aa_output_dir):
                os.makedirs(aa_output_dir)
            if not os.path.isfile(input_path):
                log.write(f"Missing file: {input_file}\n")
                print(f"Missing file: {input_file}")
                continue
            extracted_count = process_single_file(input_path, fasta_file,
                                                  aa_output_dir, organism)
            status = (f"Extracted {extracted_count} sequences" if extracted_count > 0 
                      else "No matches found")
            log.write(f"{input_file}: {status}\n")
            print(f"{input_file}: {status}")

if __name__ == "__main__":
    main()
