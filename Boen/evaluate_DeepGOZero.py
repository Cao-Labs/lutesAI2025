import os
import subprocess
import pandas as pd
from datetime import datetime
from Bio import SeqIO
import click as ck
import time
import multiprocessing

# ==============================================================================
# --- CONFIGURATION ---
# YOU MUST EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ==============================================================================

# 1. Path to your input protein sequences for prediction
FASTA_FILE = "/data/summer2020/Boen/benchmark_testing_sequences.fasta"

# 2. Path to the main InterProScan executable shell script
INTERPROSCAN_PATH = "/data/shared/tools/interproscan-5.75-106.0/interproscan.sh"

# 3. Path to the cloned DeepGOZero repository
DEEPGOZERO_PATH = "/data/shared/tools/deepgozero"

# 4. Path to the downloaded and extracted data from the DeepGOZero website
DEEPGOZERO_DATA_ROOT = "/data/shared/tools/deepgozero/data"

# 5. A directory to store all intermediate and final output files
OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_predictions"

# 6. Number of CPU cores for InterProScan
CPU_CORES = "8"

# ==============================================================================
# --- SCRIPT LOGIC (No edits needed below this line) ---
# ==============================================================================

STATUS_FILE = os.path.join(OUTPUT_DIR, "status.txt")

def update_status(message):
    """Writes a timestamped message to the status file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(STATUS_FILE, "a") as f:
        f.write(f"{now} - {message}\n")
    print(message)

def filter_long_sequences(input_fasta, output_dir, max_length=4000):
    """
    Filters a FASTA file to remove sequences longer than a specified limit.
    This prevents InterProScan from failing on exceptionally long proteins.
    """
    filtered_fasta_path = os.path.join(output_dir, "filtered_sequences.fasta")
    update_status(f"STEP 0: Filtering sequences longer than {max_length} aa from {input_fasta}...")
    
    short_sequences = []
    long_sequence_count = 0
    total_count = 0

    for record in SeqIO.parse(input_fasta, "fasta"):
        total_count += 1
        if len(record.seq) <= max_length:
            short_sequences.append(record)
        else:
            long_sequence_count += 1
            
    SeqIO.write(short_sequences, filtered_fasta_path, "fasta")
    
    if long_sequence_count > 0:
        update_status(f"STEP 0: Removed {long_sequence_count} sequences longer than {max_length} aa.")
    update_status(f"STEP 0: Proceeding with {len(short_sequences)} of {total_count} total sequences.")
    
    return filtered_fasta_path

def count_fasta_proteins(fasta_file):
    """Counts the number of sequences in a FASTA file."""
    try:
        return sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    except FileNotFoundError:
        return 0

def monitor_interproscan_progress(tsv_file, total_proteins, status_file_path):
    """A separate process to monitor the InterProScan output file."""
    start_time = time.time()
    while True:
        time.sleep(60) # Check once every minute
        if not os.path.exists(tsv_file):
            continue
        try:
            with open(tsv_file, 'r') as f:
                processed_ids = {line.split('\t')[0] for line in f}
            
            processed_count = len(processed_ids)
            percentage = (processed_count / total_proteins) * 100 if total_proteins > 0 else 0
            elapsed_time = time.time() - start_time
            
            message = (f"STEP 1: InterProScan in progress... "
                       f"Processed {processed_count}/{total_proteins} proteins ({percentage:.2f}%). "
                       f"Elapsed time: {elapsed_time/60:.1f} minutes.")
            
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(status_file_path, "w") as f:
                f.write(f"{now} - {message}\n")
        except Exception:
            continue

def run_interproscan(fasta_file, output_tsv, total_proteins):
    """Executes InterProScan with combined fixes for stability."""
    update_status("STEP 1: Starting InterProScan analysis (robust mode)...")
    
    # --- COMBINED FIX ---
    # We are both using the filtered FASTA file AND disabling fragile applications
    # to give the highest probability of success.
    cmd = [
        INTERPROSCAN_PATH,
        "-i", fasta_file,
        "-f", "TSV",
        "-o", output_tsv,
        "--goterms",
        "-cpu", CPU_CORES,
        "--disable-applications", "Coils,Phobius,SignalP"
    ]

    monitor = multiprocessing.Process(target=monitor_interproscan_progress, args=(output_tsv, total_proteins, STATUS_FILE))
    monitor.daemon = True
    monitor.start()

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            error_message = f"InterProScan failed. STDERR: {stderr}"
            update_status(f"ERROR: {error_message}")
            raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
        update_status("STEP 1: InterProScan completed successfully.")
    finally:
        monitor.terminate()
        monitor.join()

def create_prediction_dataframe(fasta_file, interpro_tsv, output_pkl):
    """Parses InterProScan results and creates the required .pkl file."""
    update_status("STEP 2: Creating Prediction-Ready DataFrame...")
    interpro_map = {}
    # Use a try-except block in case the TSV file is empty after a failed run
    try:
        with open(interpro_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                protein_id = parts[0]
                if protein_id not in interpro_map:
                    interpro_map[protein_id] = set()
                if len(parts) > 11 and parts[11].startswith("IPR"):
                    interpro_map[protein_id].add(parts[11])
    except FileNotFoundError:
        update_status(f"WARNING: InterProScan output file not found at {interpro_tsv}. Cannot create DataFrame.")
        return False


    protein_ids = [record.id for record in SeqIO.parse(fasta_file, "fasta")]
    interpros_list = [list(interpro_map.get(pid, set())) for pid in protein_ids]
    df = pd.DataFrame({'proteins': protein_ids, 'interpros': interpros_list})
    df.to_pickle(output_pkl)
    update_status("STEP 2: DataFrame created successfully.")
    return True

def parse_deepgozero_output(stdout):
    """Parses the raw stdout from DeepGOZero prediction."""
    predictions = {}
    for line in stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) == 3 and parts[1].startswith('GO:'):
            protein_id, go_term = parts[0], parts[1]
            if protein_id not in predictions:
                predictions[protein_id] = set()
            predictions[protein_id].add(go_term)
    return predictions

def run_deepgozero_prediction(input_pkl, ontology, output_dir):
    """Runs the DeepGOZero prediction script for a specific ontology."""
    update_status(f"STEP 3 ({ontology.upper()}): Running DeepGOZero prediction...")
    prediction_script = os.path.join(DEEPGOZERO_PATH, "deepgozero_predict.py")
    ontology_data_path = os.path.join(DEEPGOZERO_DATA_ROOT, ontology)
    model_file = os.path.join(ontology_data_path, "deepgozero_zero_10.th")
    terms_file = os.path.join(ontology_data_path, "terms_zero_10.pkl")
    cmd = ["python", prediction_script, "--test-data-file", input_pkl, "--model-file", model_file, "--terms-file", terms_file, "--device", "cuda:0"]
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=DEEPGOZERO_PATH)
        update_status(f"STEP 3 ({ontology.upper()}): DeepGOZero prediction completed.")
        return parse_deepgozero_output(process.stdout)
    except subprocess.CalledProcessError as e:
        error_message = f"DeepGOZero failed for {ontology.upper()}. STDERR: {e.stderr}"
        update_status(f"ERROR: {error_message}")
        raise e

def save_benchmark_output(predictions, ontology, output_dir):
    """Saves predictions in a benchmark-compatible format."""
    update_status(f"STEP 4 ({ontology.upper()}): Saving final predictions...")
    output_path = os.path.join(output_dir, f"predictions_{ontology}_benchmark.txt")
    
    lines_written = 0
    with open(output_path, 'w') as f:
        for protein_id, go_terms in sorted(predictions.items()):
            for go_term in sorted(list(go_terms)):
                f.write(f"{protein_id}\t{go_term}\n")
                lines_written += 1
    
    update_status(f"STEP 4 ({ontology.upper()}): Saved {lines_written} predictions to {output_path}")

@ck.command()
@ck.option('--force-rerun', is_flag=True, help="Force re-running InterProScan and data preparation.")
def main(force_rerun):
    """Main function to orchestrate the entire pipeline for all ontologies."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Clear status file at the beginning of a run
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
    update_status("Pipeline starting...")
    
    filtered_fasta_file = filter_long_sequences(FASTA_FILE, OUTPUT_DIR)
    
    interpro_output_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    prediction_input_pkl = os.path.join(OUTPUT_DIR, "prediction_input.pkl")
    
    total_proteins = count_fasta_proteins(filtered_fasta_file)
    if total_proteins == 0:
        update_status(f"ERROR: No proteins left after filtering. Check sequence lengths in: {FASTA_FILE}")
        return
    
    if force_rerun or not os.path.exists(interpro_output_tsv):
        run_interproscan(filtered_fasta_file, interpro_output_tsv, total_proteins)
    else:
        update_status("STEP 1: SKIPPED - Found existing InterProScan results.")
    
    if force_rerun or not os.path.exists(prediction_input_pkl):
        df_created = create_prediction_dataframe(filtered_fasta_file, interpro_output_tsv, prediction_input_pkl)
        if not df_created:
            update_status("Exiting pipeline due to missing InterProScan data.")
            return
    else:
        update_status("STEP 2: SKIPPED - Found existing prediction PKL.")

    ontologies_to_predict = ['mf', 'bp', 'cc']
    for ontology in ontologies_to_predict:
        deepgo_preds = run_deepgozero_prediction(prediction_input_pkl, ontology, OUTPUT_DIR)
        save_benchmark_output(deepgo_preds, ontology, OUTPUT_DIR)
    
    update_status("Pipeline finished successfully for all ontologies!")

if __name__ == "__main__":
    main()
