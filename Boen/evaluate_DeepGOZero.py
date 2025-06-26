import os
import subprocess
import pandas as pd
from datetime import datetime
from Bio import SeqIO
import click as ck
import time
import multiprocessing
import glob

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

def filter_long_sequences(input_fasta, output_dir, max_length=3000):
    """Filters a FASTA file to remove sequences longer than a specified limit."""
    filtered_fasta_path = os.path.join(output_dir, "filtered_sequences.fasta")
    update_status(f"STEP 0: Filtering sequences longer than {max_length} aa from {input_fasta}...")
    
    short_sequences = [record for record in SeqIO.parse(input_fasta, "fasta") if len(record.seq) <= max_length]
    original_count = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    filtered_count = len(short_sequences)
            
    SeqIO.write(short_sequences, filtered_fasta_path, "fasta")
    
    if original_count > filtered_count:
        update_status(f"STEP 0: Removed {original_count - filtered_count} sequences longer than {max_length} aa.")
    update_status(f"STEP 0: Proceeding with {filtered_count} of {original_count} total sequences.")
    
    return filtered_fasta_path

def split_fasta(input_fasta, output_dir, chunk_size=50):
    """Splits a fasta file into smaller chunks."""
    update_status(f"STEP 0.5: Splitting FASTA into chunks of {chunk_size}...")
    chunk_dir = os.path.join(output_dir, "fasta_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    record_iter = SeqIO.parse(open(input_fasta), 'fasta')
    for i, batch in enumerate(batch_iterator(record_iter, chunk_size)):
        filename = os.path.join(chunk_dir, f"chunk_{i + 1}.fasta")
        with open(filename, "w") as handle:
            count = SeqIO.write(batch, handle, "fasta")
        update_status(f"Generated chunk {i+1} with {count} sequences.")
    return chunk_dir

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size."""
    entry = True
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = None
            if entry is None:
                break
            batch.append(entry)
        if batch:
            yield batch

def run_interproscan_chunk(chunk_path, output_dir, chunk_num):
    """
    Executes InterProScan on a single FASTA chunk.
    If it fails, it logs the protein IDs in the failed chunk and continues.
    """
    chunk_name = os.path.basename(chunk_path)
    total_proteins = sum(1 for _ in SeqIO.parse(chunk_path, "fasta"))
    output_tsv = os.path.join(output_dir, f"{os.path.splitext(chunk_name)[0]}.tsv")

    update_status(f"STEP 1.{chunk_num}: Starting InterProScan for {chunk_name} ({total_proteins} proteins)...")
    
    cmd = [
        INTERPROSCAN_PATH, "-i", chunk_path, "-f", "TSV", "-o", output_tsv,
        "--goterms", "-cpu", CPU_CORES, "--disable-applications", "Coils,Phobius,SignalP"
    ]

    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        update_status(f"STEP 1.{chunk_num}: InterProScan completed for {chunk_name}.")
    except subprocess.CalledProcessError as e:
        # --- MODIFIED BEHAVIOR ---
        # Now identifies and logs the specific proteins in the failed chunk.
        failed_proteins = [record.id for record in SeqIO.parse(chunk_path, "fasta")]
        error_message = (
            f"WARNING: InterProScan failed on chunk {chunk_num} ({chunk_name}) and will be SKIPPED.\n"
            f"         The {len(failed_proteins)} protein(s) in this chunk are:\n"
            f"         {', '.join(failed_proteins)}\n"
            f"         Error: {e.stderr.strip()}"
        )
        update_status(error_message)
    return output_tsv

def merge_tsv_results(chunk_dir, final_tsv_path):
    """Merges all successfully created chunked TSV files into one."""
    update_status("STEP 1.M: Merging all InterProScan chunk results...")
    tsv_files = glob.glob(os.path.join(chunk_dir, "*.tsv"))
    if not tsv_files:
        update_status("WARNING: No successful InterProScan results to merge.")
        return

    with open(final_tsv_path, 'wb') as outfile:
        for tsv_file in sorted(tsv_files):
            with open(tsv_file, 'rb') as infile:
                outfile.write(infile.read())
    update_status(f"STEP 1.M: Merged {len(tsv_files)} chunk files into {final_tsv_path}.")

def create_prediction_dataframe(fasta_file, interpro_tsv, output_pkl):
    """Parses InterProScan results and creates the required .pkl file."""
    update_status("STEP 2: Creating Prediction-Ready DataFrame...")
    interpro_map = {}
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
        update_status(f"ERROR: Merged InterProScan output file not found at {interpro_tsv}. Cannot create DataFrame.")
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
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)
    update_status("Pipeline starting...")
    
    filtered_fasta_file = filter_long_sequences(FASTA_FILE, OUTPUT_DIR)
    
    interpro_output_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    prediction_input_pkl = os.path.join(OUTPUT_DIR, "prediction_input.pkl")

    if force_rerun or not os.path.exists(interpro_output_tsv):
        chunk_dir = split_fasta(filtered_fasta_file, OUTPUT_DIR)
        chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "*.fasta")))
        
        for i, chunk_file in enumerate(chunk_files):
            run_interproscan_chunk(chunk_file, chunk_dir, i + 1)
        
        merge_tsv_results(chunk_dir, interpro_output_tsv)
    else:
        update_status("STEP 1: SKIPPED - Found existing merged InterProScan results.")
    
    if not os.path.exists(interpro_output_tsv):
        update_status("ERROR: No InterProScan results were generated. Exiting pipeline.")
        return

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
