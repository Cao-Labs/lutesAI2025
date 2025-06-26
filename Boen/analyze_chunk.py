import os
import re
from Bio import SeqIO
import click as ck

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# 1. Path to the specific chunk file you want to analyze.
TARGET_CHUNK_FILE = "/data/summer2020/Boen/deepgozero_predictions/fasta_chunks/chunk_131.fasta"

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

def find_longest_homopolymer(sequence):
    """Finds the longest run of a single character in a sequence."""
    max_run = 0
    if not sequence:
        return 0
    
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    max_run = max(max_run, current_run)
    return max_run

def find_invalid_chars(sequence):
    """Finds any characters that are not standard amino acids."""
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    invalid_chars = set()
    for char in sequence.upper():
        if char not in standard_aa:
            invalid_chars.add(char)
    return list(invalid_chars)

@ck.command()
def main():
    """
    Analyzes a single FASTA chunk file to find potential reasons for
    InterProScan failure.
    """
    if not os.path.exists(TARGET_CHUNK_FILE):
        print(f"ERROR: Target file not found at {TARGET_CHUNK_FILE}")
        return

    chunk_name = os.path.basename(TARGET_CHUNK_FILE)
    print(f"--- Analyzing Proteins in: {chunk_name} ---")
    
    for record in SeqIO.parse(TARGET_CHUNK_FILE, "fasta"):
        seq = str(record.seq)
        seq_len = len(seq)
        
        # --- Perform Checks ---
        invalid_chars = find_invalid_chars(seq)
        longest_run = find_longest_homopolymer(seq)

        # --- Report Findings ---
        potential_issues = []
        if invalid_chars:
            potential_issues.append(f"Contains invalid characters: {', '.join(invalid_chars)}")
        if longest_run > 20: # A long run is a potential low-complexity region
            potential_issues.append(f"Longest single amino acid run: {longest_run}")

        if potential_issues:
            print(f"  - Protein: {record.id} (Length: {seq_len})")
            for issue in potential_issues:
                print(f"    * POTENTIAL ISSUE: {issue}")
        else:
            print(f"  - Protein: {record.id} (Length: {seq_len}) - No obvious issues found.")
    print("-" * (len(chunk_name) + 20) + "\n")

if __name__ == "__main__":
    main()
