import os
import random
import shutil

# --- Configuration ---
# Source directory containing the many individual .fasta files
SRC_SEQ_DIR = "/data/summer2020/naufal/testing_sequences"

# Source master FASTA file containing all sequences
SRC_MASTER_FASTA = os.path.join(SRC_SEQ_DIR, "testing_sequences.fasta")

# Destination directory for the new benchmark set
DEST_SEQ_DIR = "/data/summer2020/Boen/benchmark_testing_sequences"

# Destination master FASTA file for the benchmark set
DEST_MASTER_FASTA = os.path.join(DEST_SEQ_DIR, "benchmark_sequences.fasta")

# Number of sequences to select for the benchmark set
NUM_SAMPLES = 10000

# --- Script Start ---

def create_benchmark_dataset():
    """
    Selects a random sample of sequence files and creates a new benchmark dataset,
    including a new master FASTA file.
    """
    print("--- Starting Benchmark Dataset Creation ---")

    # Step 1: Create the destination directory
    print(f"Creating destination directory: {DEST_SEQ_DIR}")
    os.makedirs(DEST_SEQ_DIR, exist_ok=True)

    # Step 2: Get a list of all .fasta files in the source directory
    try:
        # We filter to ensure we only get .fasta files and not the master file itself
        all_files = [f for f in os.listdir(SRC_SEQ_DIR) if f.endswith('.fasta') and f != "testing_sequences.fasta"]
        if len(all_files) < NUM_SAMPLES:
            print(f"Error: Source directory contains {len(all_files)} files, which is less than the requested {NUM_SAMPLES} samples.")
            return
        print(f"Found {len(all_files)} individual sequence files in source directory.")
    except FileNotFoundError:
        print(f"Error: Source directory not found at {SRC_SEQ_DIR}")
        return

    # Step 3: Randomly select 10,000 files
    random.seed(42) # Use a seed for reproducibility
    selected_files = random.sample(all_files, NUM_SAMPLES)
    print(f"Randomly selected {len(selected_files)} files for the benchmark set.")

    # We need the protein IDs without the '.fasta' extension for later
    selected_ids = {os.path.splitext(f)[0] for f in selected_files}

    # Step 4: Copy the selected individual .fasta files
    print(f"Copying {len(selected_files)} files to {DEST_SEQ_DIR}...")
    copied_count = 0
    for filename in selected_files:
        src_path = os.path.join(SRC_SEQ_DIR, filename)
        dest_path = os.path.join(DEST_SEQ_DIR, filename)
        shutil.copy2(src_path, dest_path)
        copied_count += 1
    print(f"Successfully copied {copied_count} files.")

    # Step 5: Create the new master FASTA file for the benchmark set
    print(f"Creating new master FASTA file at {DEST_MASTER_FASTA}...")
    sequences_written = 0
    try:
        with open(SRC_MASTER_FASTA, "r") as infile, open(DEST_MASTER_FASTA, "w") as outfile:
            current_id, seq_lines = None, []
            for line in infile:
                if line.startswith(">"):
                    # If we have a stored ID, check if it should be written
                    if current_id and current_id in selected_ids:
                        outfile.write(f">{current_id}\n")
                        outfile.write("".join(seq_lines))
                        sequences_written += 1

                    # Start processing the new ID
                    current_id = line.strip()[1:]
                    seq_lines = []
                else:
                    # Append sequence lines if we are inside a record
                    if current_id:
                        seq_lines.append(line)
            
            # Write the very last sequence in the file if it's in our set
            if current_id and current_id in selected_ids:
                outfile.write(f">{current_id}\n")
                outfile.write("".join(seq_lines))
                sequences_written += 1

        print(f"Wrote {sequences_written} sequences to the new master file.")

    except FileNotFoundError:
        print(f"Error: Master FASTA file not found at {SRC_MASTER_FASTA}")
        return

    print("\n--- Benchmark Dataset Creation Complete ---")
    print(f"Total files copied: {copied_count}")
    print(f"Total sequences in new master file: {sequences_written}")

if __name__ == "__main__":
    create_benchmark_dataset()

