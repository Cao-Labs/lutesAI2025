import sys
import os

def load_ground_truth(ground_truth_file):
    """
    Loads the ground truth file into a dictionary for fast lookups.

    Args:
        ground_truth_file (str): Path to the matched_ids_with_go.txt file.

    Returns:
        dict: A dictionary mapping protein IDs to their GO terms.
              Example: {'001R_FRG3G': 'GO:0046782', ...}
    """
    print(f"Loading ground truth data from: {ground_truth_file}")
    ground_truth_map = {}
    try:
        with open(ground_truth_file, 'r') as f:
            for line in f:
                # The file might be separated by tabs or multiple spaces
                parts = line.strip().split()
                if len(parts) >= 2:
                    protein_id = parts[0]
                    go_terms = parts[1]
                    ground_truth_map[protein_id] = go_terms
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        sys.exit(1)
    
    print(f"Loaded {len(ground_truth_map)} ground truth entries.")
    return ground_truth_map

# In your ground truth finder script...

def extract_go_terms_from_directory(input_dir, ground_truth_map, output_file):
    """
    Reads all FASTA files from a directory, uses the filename as the protein ID,
    looks up GO terms, and saves the results to a single output file.
    """
    print(f"Processing FASTA files from directory: {input_dir}")
    
    found_count = 0
    not_found_count = 0
    
    # Ensure the output directory exists
    output_dir_path = os.path.dirname(output_file)
    os.makedirs(output_dir_path, exist_ok=True)

    with open(output_file, 'w') as fout:
        # --- FIX 1: DO NOT WRITE A HEADER ---
        # fout.write("Protein_ID\tGround_Truth_GO_Terms\n") 

        # Loop through all files in the input directory
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".fasta"):
                protein_id = os.path.splitext(filename)[0]
                
                if protein_id in ground_truth_map:
                    # --- FIX 2: SPLIT THE GO TERMS AND LOOP THROUGH THEM ---
                    go_terms_string = ground_truth_map[protein_id]
                    
                    # Split by semicolon, which is common for GO annotations
                    individual_go_terms = go_terms_string.split(';')
                    
                    for go_term in individual_go_terms:
                        if go_term: # Ensure it's not an empty string
                            fout.write(f"{protein_id}\t{go_term.strip()}\n")
                    # --------------------------------------------------------
                    found_count += 1
                else:
                    # This part is fine, but you should address the ID mismatch issue
                    # so fewer proteins end up here.
                    print(f"ID not found in ground truth map: {protein_id}")
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
