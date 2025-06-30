import sys
import os
from collections import defaultdict

def load_ground_truth(ground_truth_file):
    """
    Loads the ground truth file into a dictionary for fast lookups.
    Now handles multiple GO terms per protein.

    Args:
        ground_truth_file (str): Path to the matched_ids_with_go.txt file.

    Returns:
        dict: A dictionary mapping protein IDs to their GO terms (as sets).
              Example: {'001R_FRG3G': {'GO:0046782', 'GO:0008150'}, ...}
    """
    print(f"Loading ground truth data from: {ground_truth_file}")
    ground_truth_map = defaultdict(set)
    
    try:
        with open(ground_truth_file, 'r') as f:
            for line in f:
                # The file might be separated by tabs or multiple spaces
                parts = line.strip().split()
                if len(parts) >= 2:
                    protein_id = parts[0]
                    go_term = parts[1]
                    # Add GO term to the set for this protein
                    ground_truth_map[protein_id].add(go_term)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        sys.exit(1)
    
    print(f"Loaded {len(ground_truth_map)} unique proteins.")
    
    # Print some statistics
    total_go_terms = sum(len(terms) for terms in ground_truth_map.values())
    avg_terms_per_protein = total_go_terms / len(ground_truth_map) if ground_truth_map else 0
    print(f"Total GO term annotations: {total_go_terms}")
    print(f"Average GO terms per protein: {avg_terms_per_protein:.2f}")
    
    return ground_truth_map

def extract_go_terms_from_directory(input_dir, ground_truth_map, output_file):
    """
    Reads all FASTA files from a directory, uses the filename as the protein ID,
    looks up GO terms, and saves the results to a single output file.
    Now properly handles multiple GO terms per protein.

    Args:
        input_dir (str): Path to the directory containing FASTA files.
        ground_truth_map (dict): The dictionary of protein IDs and their GO terms (sets).
        output_file (str): Path to the file where results will be saved.
    """
    print(f"Processing FASTA files from directory: {input_dir}")
    
    found_count = 0
    not_found_count = 0
    
    # Ensure the output directory exists
    output_dir_path = os.path.dirname(output_file)
    os.makedirs(output_dir_path, exist_ok=True)

    with open(output_file, 'w') as fout:
        fout.write("Protein_ID\tGround_Truth_GO_Terms\n") # Write a header

        # Loop through all files in the input directory
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".fasta"):
                # Extract protein ID from the filename (e.g., "PROTEIN_ID.fasta")
                protein_id = os.path.splitext(filename)[0]
                
                # Look up the ID in our ground truth dictionary
                if protein_id in ground_truth_map:
                    go_terms_set = ground_truth_map[protein_id]
                    # Join multiple GO terms with semicolons
                    go_terms_str = ';'.join(sorted(go_terms_set))
                    fout.write(f"{protein_id}\t{go_terms_str}\n")
                    found_count += 1
                else:
                    # Handle cases where the protein ID is not in the ground truth file
                    fout.write(f"{protein_id}\tNOT_FOUND\n")
                    not_found_count += 1
    
    print("\nProcessing Complete.")
    print(f"Results saved to: {output_file}")
    print(f" - Found GO terms for {found_count} proteins.")
    print(f" - Could not find GO terms for {not_found_count} proteins.")


if __name__ == "__main__":
    # --- Define the paths directly in the script ---
    FASTA_DIR = "/data/summer2020/Boen/benchmark_testing_sequences"
    GROUND_TRUTH_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"
    OUTPUT_DIR = "/data/summer2020/Boen/ground_truth_go_terms"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "consolidated_ground_truth.tsv")

    print("--- Starting Ground Truth Extraction ---")
    print(f"Input FASTA Directory: {FASTA_DIR}")
    print(f"Ground Truth File: {GROUND_TRUTH_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print("--------------------------------------\n")

    # 1. Load the ground truth data into memory
    truth_data = load_ground_truth(GROUND_TRUTH_FILE)

    # 2. Process the FASTA directory and write the results
    extract_go_terms_from_directory(FASTA_DIR, truth_data, OUTPUT_FILE)