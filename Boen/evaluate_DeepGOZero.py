import os
import subprocess
import pandas as pd
from datetime import datetime
from Bio import SeqIO
import click as ck
import time               # --- NEW ---
import multiprocessing    # --- NEW ---

# ==============================================================================
# --- CONFIGURATION ---
# YOU MUST EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ==============================================================================

# 1. Path to your input protein sequences
FASTA_FILE = "/data/summer2020/naufal/testing_sequences.fasta"

# 2. Path to the main InterProScan executable shell script
INTERPROSCAN_PATH = "/home/lutesAI2025/tools/interproscan-5.75-106.0/interproscan.sh"

# 3. Path to the cloned DeepGOZero repository
DEEPGOZERO_PATH = "/data/shared/tools/deepgozero"

# 4. Path to the downloaded and extracted data from the DeepGOZero website
DEEPGOZERO_DATA_ROOT = "/data/shared/tools/deepgozero/data"

# 5. A directory to store all intermediate and final output files
OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_pipeline_output"

# 6. Which ontology to predict for: 'mf', 'bp', or 'cc'
ONTOLOGY = 'bp'

# 7. Number of CPU cores for InterProScan
CPU_CORES = "8"

# ==============================================================================
# --- SCRIPT LOGIC (No edits needed below this line) ---
# ==============================================================================

STATUS_FILE = os.path.join(OUTPUT_DIR, "status.txt")

# --- NEW ---
def update_status(message):
    """Writes a timestamped message to the status file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(STATUS_FILE, "w") as f:
        f.write(f"{now} - {message}\n")
    # Also print to the main log for debugging
    print(message)

# --- NEW ---
def count_fasta_proteins(fasta_file):
    """Counts the number of sequences in a FASTA file."""
    try:
        return sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    except FileNotFoundError:
        return 0

# --- NEW ---
def monitor_interproscan_progress(tsv_file, total_proteins, status_file_path):
    """A separate process to monitor the InterProScan output file."""
    while True:
        time.sleep(60) # Check once every minute
        if not os.path.exists(tsv_file):
            continue
        try:
            # Efficiently count unique proteins processed so far
            with open(tsv_file, 'r') as f:
                processed_ids = {line.split('\t')[0] for line in f}
            
            processed_count = len(processed_ids)
            percentage = (processed_count / total_proteins) * 100
            
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = (f"STEP 1: InterProScan in progress... "
                       f"Processed {processed_count} / {total_proteins} proteins ({percentage:.2f}%)")
            with open(status_file_path, "w") as f:
                f.write(f"{now} - {message}\n")
        except Exception as e:
            # If there's an error reading the file (e.g., it's being written), just skip
            continue


def run_interproscan(fasta_file, output_tsv, total_proteins):
    """Executes InterProScan and starts a parallel monitoring process."""
    update_status("STEP 1: Starting InterProScan analysis...")
    
    cmd = [
        INTERPROSCAN_PATH, "-i", fasta_file, "-f", "TSV", "-o", output_tsv,
        "--goterms", "-cpu", CPU_CORES
    ]

    # --- NEW: Start the monitor process ---
    monitor = multiprocessing.Process(
        target=monitor_interproscan_progress,
        args=(output_tsv, total_proteins, STATUS_FILE)
    )
    monitor.daemon = True
    monitor.start()

    try:
        # Using Popen to run InterProScan in a non-blocking way
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            # If InterProScan failed, write its error to the status file and raise exception
            error_message = f"InterProScan failed with exit code {process.returncode}. STDERR: {stderr}"
            update_status(f"ERROR: {error_message}")
            raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
        
        update_status(f"STEP 1: InterProScan completed successfully. Processed {total_proteins} proteins.")

    finally:
        # --- NEW: Terminate the monitor process ---
        monitor.terminate()
        monitor.join()

def create_prediction_dataframe(fasta_file, interpro_tsv, output_pkl):
    """Parses InterProScan results and creates the required .pkl file."""
    update_status("STEP 2: Creating Prediction-Ready DataFrame...")
    
    interpro_map = {}
    with open(interpro_tsv, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            protein_id = parts[0]
            if protein_id not in interpro_map:
                interpro_map[protein_id] = set()
            if len(parts) > 11 and parts[11].startswith("IPR"):
                interpro_map[protein_id].add(parts[11])

    protein_ids = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
    interpros_list = [list(interpro_map.get(pid, set())) for pid in protein_ids]

    df = pd.DataFrame({'proteins': protein_ids, 'interpros': interpros_list})
    df.to_pickle(output_pkl)
    update_status("STEP 2: DataFrame created successfully.")

def run_deepgozero_prediction(input_pkl, ontology, output_dir):
    """Runs the DeepGOZero prediction script and saves the results."""
    update_status("STEP 3: Running DeepGOZero prediction...")
    
    prediction_script = os.path.join(DEEPGOZERO_PATH, "deepgozero_predict.py")
    ontology_data_path = os.path.join(DEEPGOZERO_DATA_ROOT, ontology)
    
    model_file = os.path.join(ontology_data_path, "deepgozero_zero_10.th")
    terms_file = os.path.join(ontology_data_path, "terms_zero_10.pkl")
        
    cmd = [
        "python", prediction_script, "--test-data-file", input_pkl,
        "--model-file", model_file, "--terms-file", terms_file, "--device", "cuda:0"
    ]
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=DEEPGOZERO_PATH)
        update_status("STEP 3: DeepGOZero prediction completed.")
        parse_and_save_results(process.stdout, ontology, input_pkl, output_dir)
    except subprocess.CalledProcessError as e:
        error_message = f"DeepGOZero failed. STDERR: {e.stderr}"
        update_status(f"ERROR: {error_message}")
        raise e

def parse_and_save_results(stdout, ontology, input_pkl, output_dir):
    """Parses the prediction output from stdout and saves it to a CSV and summary file."""
    update_status("STEP 4: Parsing and Saving Final Predictions...")
    predictions = []
    for line in stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) == 3 and parts[1].startswith('GO:'):
            try:
                predictions.append({'Protein_ID': parts[0], 'GO_Term': parts[1], 'Score': float(parts[2])})
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(predictions)
    csv_path = os.path.join(output_dir, f"predictions_{ontology}.csv")
    df.to_csv(csv_path, index=False)
    update_status(f"STEP 4: Saved {len(df)} predictions to {csv_path}")


@ck.command()
@ck.option('--force-rerun', is_flag=True, help="Force re-running InterProScan and data preparation even if output files exist.")
def main(force_rerun):
    """Main function to orchestrate the entire pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    update_status("Pipeline starting...")
    
    interpro_output_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    prediction_input_pkl = os.path.join(OUTPUT_DIR, "prediction_input.pkl")
    
    total_proteins = count_fasta_proteins(FASTA_FILE)
    if total_proteins == 0:
        update_status(f"ERROR: No proteins found in FASTA file: {FASTA_FILE}")
        return
    update_status(f"Found {total_proteins} proteins in input file.")

    if force_rerun or not os.path.exists(interpro_output_tsv):
        run_interproscan(FASTA_FILE, interpro_output_tsv, total_proteins)
    else:
        update_status("STEP 1: SKIPPED - Found existing InterProScan results.")
    
    if force_rerun or not os.path.exists(prediction_input_pkl):
        create_prediction_dataframe(FASTA_FILE, interpro_output_tsv, prediction_input_pkl)
    else:
        update_status("STEP 2: SKIPPED - Found existing prediction PKL.")

    run_deepgozero_prediction(prediction_input_pkl, ONTOLOGY, OUTPUT_DIR)
    
    update_status("Pipeline finished successfully!")


if __name__ == "__main__":
    main()