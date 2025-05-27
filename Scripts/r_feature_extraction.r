#!/usr/bin/env Rscript
# Authors: Kavin S. Arulselvan,
#          Priyadharshini Pulavendean,
#          Vaishnavi Saravanan,
#          Ragothaman M. Yennamalli, SASTRA Deemed to be University
# Version: 1.0
# Last Modified: 2025-02-17
# Description:
#   Extracts various protein features from amino acid sequences in a FASTA file.
#   Calculates entropy measures, amino acid frequencies, physicochemical properties,
#   and network analysis metrics for each sequence.
#   Outputs a CSV file with the extracted features.
#
# Usage:
#   Rscript r_feature_extraction.R
#   (Then enter the input FASTA file name when prompted)
#
# Input:
#   A FASTA file containing amino acid sequences.
# Output:
#   <input_basename>_r.csv in the specified output directory.

library(seqinr)
library(Biostrings)
library(tibble)
library(entropy)
library(Peptides)
library(igraph)
library(tidyverse)
library(readr)

# Prompt user for input file name
input_filename <- readline(prompt = "Enter the input FASTA file name: ")

# File paths (Update these paths for your environment)
input_file <- file.path("path/to/your/directory", input_filename)
output_filename <- paste0(tools::file_path_sans_ext(input_filename), "_r.csv")
output_file <- file.path("path/to/your/directory", output_filename)  # Fixed typo

# Read the sequence from the FASTA file
seqs <- read.fasta(input_file, seqtype = "AA")

# Initialize a list to store results
result_list <- list()

# Function to calculate Shannon Entropy
calc_shannon_entropy <- function(sequence) {
  freqs <- table(sequence) / length(sequence)
  entropy <- -sum(freqs * log2(freqs), na.rm = TRUE)
  return(entropy)
}

# Function to calculate Tsallis Entropy
calc_tsallis_entropy <- function(sequence, q = 2) {
  freqs <- table(sequence) / length(sequence)
  tsallis <- (1 - sum(freqs^q)) / (q - 1)
  return(tsallis)
}

# Process each sequence
for (i in seq_along(seqs)) {
  seq <- seqs[[i]]
  seq_name <- names(seqs)[i]

  # Calculate cumulative amino acid frequencies
  valid_aa <- c("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
  aa_counts <- table(factor(seq, levels = valid_aa))
  cum_freq <- cumsum(aa_counts) / sum(aa_counts)

  # Apply Fourier Transform for cumulative frequencies
  fft_result <- fft(cum_freq)
  fft_magnitude <- Mod(fft_result)

  # Calculate Shannon and Tsallis entropy
  shannon_entropy <- calc_shannon_entropy(seq)
  tsallis_entropy <- calc_tsallis_entropy(seq)

  # Calculate k-mer frequencies (k=2) for protein sequences
  k <- 2
  seq_str <- paste(seq, collapse="")
  kmers <- table(substring(seq_str, 1:(nchar(seq_str) - k + 1), k:nchar(seq_str)))

  # Convert kmers to a named vector to match the number of rows
  kmer_vector <- rep(0, length(valid_aa))
  names(kmer_vector) <- valid_aa
  for (kmer in names(kmers)) {
    if (substr(kmer, 1, 1) %in% valid_aa && substr(kmer, 2, 2) %in% valid_aa) {
      kmer_vector[substr(kmer, 1, 1)] <- kmers[kmer]
    }
  }

  # Calculate protein charge
  seq_str_for_charge <- paste(seq, collapse="")
  charge_value <- charge(seq_str_for_charge, pH = 7.0)

  # Calculate additional properties
  mw_value <- mw(seq_str_for_charge)
  pI_value <- pI(seq_str_for_charge)
  hydrophobicity_value <- hydrophobicity(seq_str_for_charge)
  aliphatic_index <- aIndex(seq_str_for_charge)
  instability_index <- instaIndex(seq_str_for_charge)
  boman_index <- boman(seq_str_for_charge)

  # Complex network analysis
  seq_chars <- strsplit(seq_str_for_charge, "")[[1]]
  adj_matrix <- matrix(0, nrow = length(seq_chars), ncol = length(seq_chars))
  for (i in 1:(length(seq_chars)-1)) {
    for (j in (i+1):length(seq_chars)) {
      adj_matrix[i,j] <- adj_matrix[j,i] <- ifelse(seq_chars[i] == seq_chars[j], 1, 0)
    }
  }
  graph <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected")

  degree <- mean(degree(graph))
  closeness <- mean(closeness(graph))
  betweenness <- mean(betweenness(graph))
  clustering_coefficient <- transitivity(graph)

  # Combine results into a dataframe
  result_df <- tibble(
    Sequence = seq_name,
    Cumulative_Frequency = paste(cum_freq, collapse = ","),
    Fourier_Magnitude = paste(fft_magnitude, collapse = ","),
    Shannon_Entropy = shannon_entropy,
    Tsallis_Entropy = tsallis_entropy,
    Charge_pH7 = charge_value,
    MW = mw_value,
    pI = pI_value,
    Hydrophobicity = hydrophobicity_value,
    AliphaticIndex = aliphatic_index,
    InstabilityIndex = instability_index,
    BomanIndex = boman_index,
    NetworkDegree = degree,
    NetworkCloseness = closeness,
    NetworkBetweenness = betweenness,
    ClusteringCoefficient = clustering_coefficient
  )

  # Add k-mer frequencies as columns
  for (kmer in names(kmer_vector)) {
    result_df[[paste0("kmer_", kmer)]] <- kmer_vector[kmer]
  }

  # Append to result list
  result_list[[seq_name]] <- result_df
}

# Combine all results and write to CSV
final_df <- do.call(rbind, result_list)
readr::write_csv(final_df, file = output_file)

cat("Protein feature extraction completed!\nResults saved to", output_file)
