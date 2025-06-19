import os

# Input files
FASTA_FILE = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
FEATURES_FILE = "/data/summer2020/naufal/protein_features.txt"

# Read sequences into a dict: ID -> sequence
print("Reading FASTA file...")
sequences = {}
with open(FASTA_FILE, "r") as f:
    current_id = None
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            current_id = line[1:]
        elif current_id:
            sequences[current_id] = line

print(f"Loaded {len(sequences)} sequences.\n")

# Compare with features
print("Checking feature lengths...")
mismatches = 0
with open(FEATURES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue  # Skip malformed lines

        protein_id, ss, rsa = parts

        if protein_id not in sequences:
            continue  # Skip proteins not in sequence file

        seq_len = len(sequences[protein_id])
        ss_len = len(ss)
        rsa_len = len(rsa)

        if seq_len != ss_len or seq_len != rsa_len:
            mismatches += 1
            print(f"Mismatch: {protein_id}")
            print(f"  Sequence length: {seq_len}")
            print(f"  SS length      : {ss_len}")
            print(f"  RSA length     : {rsa_len}")

print(f"\nDone. Found {mismatches} mismatches.")
