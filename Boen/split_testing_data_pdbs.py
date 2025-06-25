import os
import shutil

# --- Configuration ---
# Source directory containing all of the original .pdb files
SRC_PDB_DIR = "/data/summer2020/naufal/testing_pdbs"

# Directory containing your 10,000 individual .fasta files
# The script uses the filenames (without extension) as the lookup key.
FASTA_DIR = "/data/summer2020/Boen/benchmark_testing_sequences"

# Destination directory for the corresponding .pdb files
DEST_PDB_DIR = "/data/summer2020/Boen/benchmark_testing_pdbs"

# --- IMPORTANT ---
# Path to the UniProt ID mapping file.
# The script assumes this file has the format: UniProtID ProteinID
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"

# --- Helper Function ---

def create_id_map_from_file(mapping_filepath):
    """
    Parses the idmapping_uni.txt file to create a mapping from the
    protein ID (e.g., '001R_FRG3G') to the UniProt ID (e.g., 'Q6GZX4').
    """
    print(f"Parsing ID mapping file to create map: {mapping_filepath}")
    mapping = {}
    try:
        with open(mapping_filepath, 'r') as f:
            for line in f:
                # Strip whitespace and split into parts
                parts = line.strip().split()
                if len(parts) >= 2:
                    uniprot_id = parts[0]
                    protein_id = parts[1]
                    # Map protein_id to uniprot_id
                    mapping[protein_id] = uniprot_id
    except FileNotFoundError:
        print(f"FATAL ERROR: ID Mapping file not found at {mapping_filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the ID mapping file: {e}")
        return None
        
    print(f"Successfully created a map with {len(mapping)} entries.")
    return mapping

# --- Main Script ---

def create_corresponding_pdb_set():
    """
    Copies PDB files corresponding to individual FASTA files by using an
    ID mapping file to link protein IDs to UniProt IDs.
    """
    print("--- Starting PDB Set Creation ---")

    # Step 1: Create the ID mapping from the provided file
    id_map = create_id_map_from_file(ID_MAPPING_FILE)
    if id_map is None or not id_map:
        print("Could not create ID map. Aborting script.")
        return

    # Step 2: Check if the destination directory exists and delete it
    if os.path.exists(DEST_PDB_DIR):
        print(f"\nFound existing directory at {DEST_PDB_DIR}. Deleting it.")
        try:
            shutil.rmtree(DEST_PDB_DIR)
            print("Successfully deleted old directory.")
        except Exception as e:
            print(f"Error: Could not delete directory {DEST_PDB_DIR}. Error: {e}")
            return

    # Step 3: Create the fresh destination directory
    print(f"Creating new destination directory: {DEST_PDB_DIR}")
    os.makedirs(DEST_PDB_DIR)

    # Step 4: Iterate through individual FASTA files and find their PDBs
    print(f"\nMatching FASTA files from {FASTA_DIR} with PDBs from {SRC_PDB_DIR}...")
    
    try:
        fasta_files = [f for f in os.listdir(FASTA_DIR) if f.endswith(('.fasta', '.fa'))]
    except FileNotFoundError:
        print(f"Error: Individual FASTA directory not found at {FASTA_DIR}")
        return

    copied_count = 0
    not_found_in_map = 0
    not_found_in_source = 0

    for fasta_filename in fasta_files:
        # Get protein ID from filename, e.g., 'SUCC_RHORT' from 'SUCC_RHORT.fasta'
        protein_id = os.path.splitext(fasta_filename)[0]
        
        # Look up the UniProt ID in our map
        uniprot_id = id_map.get(protein_id)
        
        if not uniprot_id:
            not_found_in_map += 1
            continue

        # Construct the target PDB filename
        pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
        src_path = os.path.join(SRC_PDB_DIR, pdb_filename)
        dest_path = os.path.join(DEST_PDB_DIR, pdb_filename)

        # Check if the PDB file exists and copy it
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Could not copy file {pdb_filename}. Error: {e}")
        else:
            not_found_in_source += 1

    print(f"\n--- PDB Set Creation Complete ---")
    print(f"Successfully copied {copied_count} files to {DEST_PDB_DIR}.")
    if not_found_in_map > 0:
        print(f"Warning: {not_found_in_map} FASTA filenames were not found in the ID mapping file.")
    if not_found_in_source > 0:
        print(f"Warning: {not_found_in_source} corresponding PDB files were not found in the source directory.")

if __name__ == "__main__":
    create_corresponding_pdb_set()
