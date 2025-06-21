import os
import subprocess
import pandas as pd
from datetime import datetime
from Bio import SeqIO
import click as ck

# ==============================================================================
# --- CONFIGURATION ---
# YOU MUST EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ==============================================================================

# 1. Path to your input protein sequences
FASTA_FILE = "/data/summer2020/naufal/testing_sequences.fasta"

# 2. Path to the main InterProScan executable shell script
INTERPROSCAN_PATH = "/data/shared/tools/interproscan-5.75-106.0/interproscan.sh" # e.g., /opt/interproscan/interproscan.sh

# 3. Path to the cloned DeepGOZero repository
DEEPGOZERO_PATH = "/data/shared/tools/deepgozero"

# 4. Path to the downloaded and extracted data from the DeepGOZero website
#    (This directory should contain 'go.norm' and the 'mf', 'bp', 'cc' subdirectories)
DEEPGOZERO_DATA_ROOT = "/data/shared/tools/deepgozero/data" # e.g., /data/shared/tools/deepgozero/data

# 5. A directory to store all intermediate and final output files
OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_predictions"

# 6. Which ontology to predict for: 'mf', 'bp', or 'cc'
ONTOLOGY = 'bp'  # Options: 'mf' (Molecular Function), 'bp' (Biological Process), 'cc' (Cellular Component)


# ==============================================================================
# --- SCRIPT LOGIC (No edits needed below this line) ---
# ==============================================================================

def run_interproscan(fasta_file, output_tsv):
    """Executes the InterProScan command-line tool."""
    print("--- STEP 1: Running InterProScan ---")
    print(f"Input: {fasta_file}")
    print(f"Output: {output_tsv}")
    
    if not os.path.exists(INTERPROSCAN_PATH):
        raise FileNotFoundError(f"InterProScan not found at: {INTERPROSCAN_PATH}. Please check the CONFIGURATION.")

    cmd = [
        INTERPROSCAN_PATH,
        "-i", fasta_file,
        "-f", "TSV",
        "-o", output_tsv,
        "--gpa", # Get deep functional annotations
        "--goterms" # Get GO terms
    ]

    print(f"Executing command: {' '.join(cmd)}")
    print("This may take a very long time depending on your input file size...")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("InterProScan completed successfully.")
    except subprocess.CalledProcessError as e:
        print("ERROR: InterProScan failed to run.")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e

def create_prediction_dataframe(fasta_file, interpro_tsv, output_pkl):
    """Parses InterProScan results and creates the required .pkl file."""
    print("\n--- STEP 2: Creating Prediction-Ready DataFrame ---")
    print(f"Input FASTA: {fasta_file}")
    print(f"Input TSV: {interpro_tsv}")
    print(f"Output PKL: {output_pkl}")
    
    # 1. Parse InterProScan TSV output
    interpro_map = {}
    with open(interpro_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            protein_id = parts[0]
            if protein_id not in interpro_map:
                interpro_map[protein_id] = set()
            
            # InterPro IDs are in the 12th column (index 11) if present
            if len(parts) > 11 and parts[11].startswith("IPR"):
                interpro_map[protein_id].add(parts[11])

    # 2. Read protein IDs from the original FASTA file to maintain order
    protein_ids = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
    interpros_list = [list(interpro_map.get(pid, set())) for pid in protein_ids]

    # 3. Create and save the DataFrame
    df = pd.DataFrame({'proteins': protein_ids, 'interpros': interpros_list})
    df.to_pickle(output_pkl)
    
    print(f"Successfully created and saved DataFrame for {len(df)} proteins to {output_pkl}.")

def run_deepgozero_prediction(input_pkl, ontology, output_dir):
    """Runs the DeepGOZero prediction script and saves the results."""
    print("\n--- STEP 3: Running DeepGOZero Prediction ---")
    
    prediction_script = os.path.join(DEEPGOZERO_PATH, "deepgozero_predict.py")
    ontology_data_path = os.path.join(DEEPGOZERO_DATA_ROOT, ontology)
    
    # Verify required files exist
    model_file = os.path.join(ontology_data_path, "deepgozero_zero_10.th")
    terms_file = os.path.join(ontology_data_path, "terms_zero_10.pkl")
    if not all(os.path.exists(p) for p in [prediction_script, model_file, terms_file]):
        raise FileNotFoundError("A required DeepGOZero script or data file was not found. Check DEEPGOZERO_PATH and DEEPGOZERO_DATA_ROOT.")
        
    cmd = [
        "python", prediction_script,
        "--test-data-file", input_pkl,
        "--model-file", model_file,
        "--terms-file", terms_file,
        "--device", "cpu"  # Change to "cuda:0" if you have a GPU
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=DEEPGOZERO_PATH)
        print("Prediction script executed successfully.")
        parse_and_save_results(process.stdout, ontology, input_pkl, output_dir)
    except subprocess.CalledProcessError as e:
        print("ERROR: DeepGOZero prediction script failed.")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e

def parse_and_save_results(stdout, ontology, input_pkl, output_dir):
    """Parses the prediction output from stdout and saves it to a CSV and summary file."""
    print("\n--- STEP 4: Parsing and Saving Final Predictions ---")
    predictions = []
    # The authors' predict script seems to output a mix of info and results.
    # We look for lines that match the "Protein GO:Term Score" format.
    for line in stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) == 3 and parts[1].startswith('GO:'):
            try:
                predictions.append({'Protein_ID': parts[0], 'GO_Term': parts[1], 'Score': float(parts[2])})
            except (ValueError, IndexError):
                continue
    
    if not predictions:
        print("Warning: No predictions were parsed. The prediction script might have failed or produced no output.")
        print("Full raw output:\n", stdout)
        return
        
    df = pd.DataFrame(predictions)
    csv_path = os.path.join(output_dir, f"predictions_{ontology}.csv")
    summary_path = os.path.join(output_dir, f"summary_{ontology}.txt")
    
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} predictions to {csv_path}")

    with open(summary_path, "w") as f:
        f.write(f"--- DeepGOZero Prediction Summary ---\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ontology: {ontology.upper()}\n\n")
        f.write(f"Input FASTA file: {FASTA_FILE}\n")
        f.write(f"Number of unique proteins with predictions: {df['Protein_ID'].nunique()}\n")
        f.write(f"Total number of GO term predictions: {len(df)}\n\n")
        f.write(f"Results saved to: {csv_path}\n")
    print(f"Summary saved to {summary_path}")


@ck.command()
@ck.option('--force-rerun', is_flag=True, help="Force re-running InterProScan and data preparation even if output files exist.")
def main(force_rerun):
    """Main function to orchestrate the entire pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define intermediate file paths
    interpro_output_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    prediction_input_pkl = os.path.join(OUTPUT_DIR, "prediction_input.pkl")
    
    # --- Step 1 ---
    if force_rerun or not os.path.exists(interpro_output_tsv):
        run_interproscan(FASTA_FILE, interpro_output_tsv)
    else:
        print(f"--- STEP 1: SKIPPED --- Found existing InterProScan results at {interpro_output_tsv}. Use --force-rerun to run again.")
    
    # --- Step 2 ---
    if force_rerun or not os.path.exists(prediction_input_pkl):
        create_prediction_dataframe(FASTA_FILE, interpro_output_tsv, prediction_input_pkl)
    else:
        print(f"\n--- STEP 2: SKIPPED --- Found existing prediction PKL at {prediction_input_pkl}. Use --force-rerun to run again.")

    # --- Step 3 & 4 ---
    run_deepgozero_prediction(prediction_input_pkl, ONTOLOGY, OUTPUT_DIR)
    
    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()