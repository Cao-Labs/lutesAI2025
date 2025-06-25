import os
import random
import shutil

# --- Configuration ---
# Source directory containing the many individual .pdb files
SRC_PDB_DIR = "/data/summer2020/naufal/testing_pdbs"

# Destination directory for the new benchmark set of PDBs
DEST_PDB_DIR = "/data/summer2020/Boen/benchmark_testing_pdbs"

# Number of PDB files to select for the benchmark set
NUM_SAMPLES = 10000

# --- Script Start ---

def create_pdb_benchmark_dataset():
    """
    Selects a random sample of PDB files and copies them to a new 
    benchmark directory.
    """
    print("--- Starting PDB Benchmark Dataset Creation ---")

    # Step 1: Create the destination directory if it doesn't exist
    print(f"Ensuring destination directory exists: {DEST_PDB_DIR}")
    os.makedirs(DEST_PDB_DIR, exist_ok=True)

    # Step 2: Get a list of all .pdb files in the source directory
    try:
        # We filter to ensure we only get files ending with .pdb
        all_files = [f for f in os.listdir(SRC_PDB_DIR) if f.endswith('.pdb')]
        
        if not all_files:
            print(f"Error: No .pdb files found in the source directory: {SRC_PDB_DIR}")
            return
            
        print(f"Found {len(all_files)} individual PDB files in source directory.")

        if len(all_files) < NUM_SAMPLES:
            print(f"Warning: Source directory contains {len(all_files)} files, which is less than the requested {NUM_SAMPLES} samples.")
            print("The script will copy all available files.")
            num_to_select = len(all_files)
        else:
            num_to_select = NUM_SAMPLES

    except FileNotFoundError:
        print(f"Error: Source directory not found at {SRC_PDB_DIR}")
        return

    # Step 3: Randomly select the specified number of files
    random.seed(42) # Use a seed for reproducibility
    selected_files = random.sample(all_files, num_to_select)
    print(f"Randomly selected {len(selected_files)} files for the benchmark set.")

    # Step 4: Copy the selected individual .pdb files
    print(f"Copying {len(selected_files)} files to {DEST_PDB_DIR}...")
    copied_count = 0
    for filename in selected_files:
        src_path = os.path.join(SRC_PDB_DIR, filename)
        dest_path = os.path.join(DEST_PDB_DIR, filename)
        try:
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"Could not copy file {filename}. Error: {e}")

    print(f"Successfully copied {copied_count} files.")

    print("\n--- PDB Benchmark Dataset Creation Complete ---")
    print(f"Total files copied to {DEST_PDB_DIR}: {copied_count}")

if __name__ == "__main__":
    create_pdb_benchmark_dataset()
