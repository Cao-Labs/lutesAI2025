import os
import subprocess
import pandas as pd
from datetime import datetime
from Bio import SeqIO
import click as ck

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

FASTA_FILE = "/data/summer2020/Boen/benchmark_testing_sequences.fasta"
DEEPGOZERO_PATH = "/data/shared/tools/deepgozero"
DEEPGOZERO_DATA_ROOT = "/data/shared/tools/deepgozero/data"
OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_pipeline_output"
ONTOLOGY = 'bp'

STATUS_FILE = os.path.join(OUTPUT_DIR, "deepgozero_status.txt")

def update_status(message):
    """Writes a timestamped message to the status file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(STATUS_FILE, "a") as f:
        f.write(f"{now} - {message}\n")
    print(message)

def create_prediction_dataframe_fixed(fasta_file, interpro_tsv, output_pkl):
    """Creates the DataFrame in the exact format expected by deepgozero_predict.py"""
    update_status("Creating prediction-ready DataFrame (FIXED FORMAT)...")
    
    # Parse InterProScan results
    interpro_map = {}
    proteins_in_interpro = set()
    
    with open(interpro_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 12:
                continue
                
            protein_id = parts[0]
            proteins_in_interpro.add(protein_id)
            
            if protein_id not in interpro_map:
                interpro_map[protein_id] = set()
            
            # Column 11 contains InterPro accession
            if len(parts) > 11 and parts[11] and parts[11].startswith("IPR"):
                interpro_map[protein_id].add(parts[11])
    
    # Load protein IDs from FASTA
    all_protein_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        all_protein_ids.append(record.id)
    
    # Filter to proteins that have InterProScan results
    protein_ids = [pid for pid in all_protein_ids if pid in proteins_in_interpro]
    
    # Create the DataFrame in the EXACT format expected by deepgozero_predict.py
    proteins_list = []
    interpros_list = []
    prop_annotations_list = []  # This is required even though we don't have annotations
    
    for pid in protein_ids:
        proteins_list.append(pid)
        interpro_domains = list(interpro_map.get(pid, set()))
        interpros_list.append(interpro_domains)
        prop_annotations_list.append([])  # Empty list since we don't have ground truth
    
    # Create DataFrame exactly as expected by the script
    df = pd.DataFrame({
        'proteins': proteins_list,
        'interpros': interpros_list,
        'prop_annotations': prop_annotations_list  # Required by the script
    })
    
    df.to_pickle(output_pkl)
    update_status(f"Created DataFrame with {len(df)} proteins in correct format")
    
    return len(df)

def validate_deepgozero_setup_fixed(ontology):
    """Validates DeepGOZero setup with correct file names."""
    update_status("Validating DeepGOZero setup...")
    
    # Check prediction script
    prediction_script = os.path.join(DEEPGOZERO_PATH, "deepgozero_predict.py")
    if not os.path.exists(prediction_script):
        raise FileNotFoundError(f"DeepGOZero prediction script not found: {prediction_script}")
    
    # Check ontology data directory
    ontology_data_path = os.path.join(DEEPGOZERO_DATA_ROOT, ontology)
    if not os.path.exists(ontology_data_path):
        raise FileNotFoundError(f"Data directory not found: {ontology_data_path}")
    
    # Check files with CORRECT names as used in deepgozero_predict.py
    model_file = os.path.join(ontology_data_path, "deepgozero.th")
    terms_file = os.path.join(ontology_data_path, "terms.pkl")
    go_file = os.path.join(DEEPGOZERO_DATA_ROOT, "go.obo")
    
    missing_files = []
    if not os.path.exists(model_file):
        missing_files.append(model_file)
    if not os.path.exists(terms_file):
        missing_files.append(terms_file)
    if not os.path.exists(go_file):
        missing_files.append(go_file)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    update_status("DeepGOZero setup validation successful")
    return prediction_script

def run_deepgozero_prediction_fixed(input_pkl, ontology, output_dir):
    """Runs DeepGOZero with the correct parameters."""
    update_status(f"Running DeepGOZero prediction for ontology: {ontology}")
    
    prediction_script = validate_deepgozero_setup_fixed(ontology)
    
    # Use the EXACT parameters as defined in deepgozero_predict.py
    cmd = [
        "python", prediction_script,
        "--data-root", DEEPGOZERO_DATA_ROOT,
        "--ont", ontology,
        "--data-file", input_pkl,
        "--device", "cpu"
    ]
    
    update_status(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=DEEPGOZERO_PATH,
            timeout=1800
        )
        
        update_status("DeepGOZero prediction completed successfully")
        return parse_and_save_results(process.stdout, ontology, output_dir)
        
    except subprocess.CalledProcessError as e:
        update_status(f"ERROR: {e}")
        update_status(f"STDERR: {e.stderr}")
        update_status(f"STDOUT: {e.stdout}")
        raise e

def parse_and_save_results(stdout, ontology, output_dir):
    """Parse results from stdout and save to CSV."""
    update_status("Parsing and saving predictions...")
    
    predictions = []
    
    for line in stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) == 3 and parts[1].startswith('GO:'):
            try:
                predictions.append({
                    'Protein_ID': parts[0],
                    'GO_Term': parts[1],
                    'Score': float(parts[2])
                })
            except (ValueError, IndexError):
                continue
    
    if not predictions:
        update_status("WARNING: No predictions found")
        return 0
    
    # Save results
    df = pd.DataFrame(predictions)
    csv_path = os.path.join(output_dir, f"predictions_{ontology}.csv")
    df.to_csv(csv_path, index=False)
    
    update_status(f"Saved {len(df)} predictions to {csv_path}")
    update_status(f"Unique proteins: {df['Protein_ID'].nunique()}")
    update_status(f"Unique GO terms: {df['GO_Term'].nunique()}")
    update_status(f"Score range: {df['Score'].min():.4f} to {df['Score'].max():.4f}")
    
    return len(df)

@ck.command()
@ck.option('--ontology', default=ONTOLOGY, type=ck.Choice(['mf', 'bp', 'cc']))
@ck.option('--force-rerun', is_flag=True)
def main(ontology, force_rerun):
    """Fixed DeepGOZero prediction pipeline."""
    
    with open(STATUS_FILE, "w") as f:
        f.write("")
    
    update_status("=== FIXED DeepGOZero Pipeline Starting ===")
    update_status(f"Ontology: {ontology}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # File paths
    interpro_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    prediction_pkl = os.path.join(OUTPUT_DIR, f"prediction_input_{ontology}_fixed.pkl")
    prediction_csv = os.path.join(OUTPUT_DIR, f"predictions_{ontology}.csv")
    
    if not force_rerun and os.path.exists(prediction_csv):
        update_status("Predictions already exist. Use --force-rerun to regenerate.")
        return
    
    try:
        # Step 1: Validate InterProScan results exist
        if not os.path.exists(interpro_tsv):
            raise FileNotFoundError(f"InterProScan results not found: {interpro_tsv}")
        
        # Step 2: Create properly formatted DataFrame
        if force_rerun or not os.path.exists(prediction_pkl):
            protein_count = create_prediction_dataframe_fixed(FASTA_FILE, interpro_tsv, prediction_pkl)
        else:
            df = pd.read_pickle(prediction_pkl)
            protein_count = len(df)
            update_status(f"Using existing prediction data: {protein_count} proteins")
        
        # Step 3: Run DeepGOZero
        prediction_count = run_deepgozero_prediction_fixed(prediction_pkl, ontology, OUTPUT_DIR)
        
        # Summary
        update_status("=" * 50)
        update_status("FIXED PIPELINE COMPLETED!")
        update_status(f"Proteins processed: {protein_count}")
        update_status(f"Predictions generated: {prediction_count}")
        update_status("=" * 50)
        
    except Exception as e:
        update_status(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()