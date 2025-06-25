import os
import shutil

# --- Configuration ---
# The directory where your 10,000 individual .fasta files are located.
BENCHMARK_DIR = "/data/summer2020/Boen/benchmark_testing_sequences"

# The full path for the new master file you want to create.
# Note this is in the parent 'Boen' directory.
OUTPUT_MASTER_FILE = "/data/summer2020/Boen/benchmark_testing_sequences.fasta"


# --- Script Start ---

def combine_fasta_files():
    """
    Combines all .fasta files from a directory into a single master FASTA file.
    """
    print("--- Starting FASTA File Combination ---")

    # Step 1: Check if the source directory exists
    if not os.path.isdir(BENCHMARK_DIR):
        print(f"Error: Source directory not found at '{BENCHMARK_DIR}'")
        return

    # Step 2: Get a list of all .fasta files to be combined
    try:
        fasta_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith('.fasta')]
        if not fasta_files:
            print(f"Error: No .fasta files were found in '{BENCHMARK_DIR}'")
            return
        print(f"Found {len(fasta_files)} individual .fasta files to combine.")
    except Exception as e:
        print(f"An error occurred while reading the directory: {e}")
        return

    # Step 3: Combine the files
    print(f"Combining files into '{OUTPUT_MASTER_FILE}'...")
    files_combined = 0
    try:
        with open(OUTPUT_MASTER_FILE, 'wb') as outfile:
            for filename in fasta_files:
                filepath = os.path.join(BENCHMARK_DIR, filename)
                try:
                    with open(filepath, 'rb') as infile:
                        # Copy the content from the small file to the large master file
                        shutil.copyfileobj(infile, outfile)
                    files_combined += 1
                except Exception as e:
                    print(f"Warning: Could not read file '{filename}'. Skipping. Error: {e}")

    except IOError as e:
        print(f"Error: Could not write to output file '{OUTPUT_MASTER_FILE}'. Error: {e}")
        return

    print("\n--- Combination Complete ---")
    print(f"Successfully combined {files_combined} files into the master file.")


if __name__ == "__main__":
    combine_fasta_files()

