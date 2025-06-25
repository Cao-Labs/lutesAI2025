import os
import shutil

# --- Configuration ---
# Source directory containing all of the original .pdb files
SRC_PDB_DIR = "/data/summer2020/naufal/testing_pdbs"

# Directory containing your 10,000 individual .fasta files
# The script will use the names of these files to find the matching PDBs.
FASTA_DIR = "/data/summer2020/Boen/benchmark_testing_sequences"

# Destination directory for the corresponding .pdb files
DEST_PDB_DIR = "/data/summer2020/Boen/benchmark_testing_pdbs"

# --- Helper Function ---

def get_uniprot_id_from_fasta(filepath):
    """
    Parses a FASTA file to extract the UniProt ID from its header.
    It assumes a standard UniProt format in the header, like:
    >sp|Q9Y6I3|DAPB_METMJ ...
    where 'Q9Y6I3' is the ID.
    """
    try:
        with open(filepath, 'r') as f:
            # Use .strip() to remove leading/trailing whitespace and newlines
            header = f.readline().strip()
            if header.startswith('>'):
                # The UniProt ID is typically the second item when split by '|'
                parts = header.split('|')
                if len(parts) > 1:
                    uniprot_id = parts[1]
                    # Add a simple validation check for what a UniProt ID looks like
                    # It allows for hyphens in isoform IDs (e.g., P12345-1)
                    if (6 <= len(uniprot_id) <= 11) and uniprot_id.replace('-', '').isalnum():
                        return uniprot_id
    except Exception as e:
        # This will report if a file can't be opened or read
        print(f"Warning: Could not read file {os.path.basename(filepath)}. Error: {e}")
    return None

# --- Main Script ---

def create_corresponding_pdb_set():
    """
    Finds and copies PDB files that correspond to FASTA files by extracting
    UniProt IDs from the FASTA file headers. Deletes the destination
    directory if it already exists and adds debugging for failed parsing.
    """
    print("--- Starting PDB Set Creation ---")

    # Step 1: Check if the destination directory exists and delete it
    if os.path.exists(DEST_PDB_DIR):
        print(f"Found existing directory at {DEST_PDB_DIR}. Deleting it.")
        try:
            shutil.rmtree(DEST_PDB_DIR)
            print("Successfully deleted old directory.")
        except Exception as e:
            print(f"Error: Could not delete directory {DEST_PDB_DIR}. Please remove it manually. Error: {e}")
            return

    # Step 2: Create the fresh destination directory
    print(f"Creating new destination directory: {DEST_PDB_DIR}")
    os.makedirs(DEST_PDB_DIR)

    # Step 3: Get the list of target PDB filenames by parsing FASTA headers
    print(f"\nParsing FASTA files in {FASTA_DIR} to find target UniProt IDs...")
    target_pdb_filenames = set()
    failed_parses = []
    
    try:
        fasta_files = [f for f in os.listdir(FASTA_DIR) if f.endswith(('.fasta', '.fa'))]
        if not fasta_files:
            print(f"Error: No .fasta files found in your FASTA directory: {FASTA_DIR}")
            return
    except FileNotFoundError:
        print(f"Error: FASTA directory not found at {FASTA_DIR}")
        return

    # Iterate through each FASTA file to find its UniProt ID
    for fasta_filename in fasta_files:
        fasta_filepath = os.path.join(FASTA_DIR, fasta_filename)
        uniprot_id = get_uniprot_id_from_fasta(fasta_filepath)
        
        if uniprot_id:
            pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
            target_pdb_filenames.add(pdb_filename)
        else:
            # If parsing failed, save the filename to report later
            failed_parses.append(fasta_filename)

    print(f"Successfully constructed {len(target_pdb_filenames)} target PDB filenames.")

    # Step 4: Add debugging for any files that could not be parsed
    if failed_parses:
        print(f"\n--- DEBUGGING INFORMATION ---")
        print(f"Warning: Could not extract a valid UniProt ID from {len(failed_parses)} FASTA files.")
        print("This is likely because the FASTA header format is not what the script expects (e.g., >sp|ID|...).")
        print("Here are the headers from the first 5 files that failed to parse:")
        
        for filename in failed_parses[:5]:
            filepath = os.path.join(FASTA_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    header = f.readline().strip()
                    print(f"  - File: {filename}, Header: {header}")
            except Exception as e:
                print(f"  - Could not read header from {filename}. Error: {e}")
        print("-----------------------------\n")

    # Step 5: Get a list of all available PDB files from the source directory
    try:
        available_pdb_files = {f for f in os.listdir(SRC_PDB_DIR) if f.endswith('.pdb')}
        print(f"Found {len(available_pdb_files)} PDB files in the source directory.")
    except FileNotFoundError:
        print(f"Error: Source PDB directory not found at {SRC_PDB_DIR}")
        return

    # Step 6: Find the intersection of target files and available files
    pdbs_to_copy = target_pdb_filenames.intersection(available_pdb_files)
    missing_pdbs = target_pdb_filenames.difference(available_pdb_files)
    print(f"Found {len(pdbs_to_copy)} matching PDB files to copy.")

    # Step 7: Copy the matching files
    copied_count = 0
    for pdb_filename in pdbs_to_copy:
        src_path = os.path.join(SRC_PDB_DIR, pdb_filename)
        dest_path = os.path.join(DEST_PDB_DIR, pdb_filename)
        
        try:
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"Error: Could not copy file {pdb_filename}. Error: {e}")

    print(f"\n--- PDB Set Creation Complete ---")
    print(f"Successfully copied {copied_count} files to {DEST_PDB_DIR}.")

    # Step 8: Report any target PDBs that were not found in the source directory
    if missing_pdbs:
        print(f"\nWarning: Could not find {len(missing_pdbs)} corresponding PDB files in {SRC_PDB_DIR}.")

if __name__ == "__main__":
    create_corresponding_pdb_set()
