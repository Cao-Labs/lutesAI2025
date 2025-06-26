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

# 1. Path to your input protein sequences (same as used for InterProScan)
FASTA_FILE = "/data/summer2020/naufal/testing_sequences.fasta"

# 2. Path to the cloned DeepGOZero repository
DEEPGOZERO_PATH = "/data/shared/tools/deepgozero"

# 3. Path to the downloaded and extracted data from the DeepGOZero website
DEEPGOZERO_DATA_ROOT = "/data/shared/tools/deepgozero/data"

# 4. A directory where InterProScan results and other outputs are stored
OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_pipeline_output"

# 5. Which ontology to predict for: 'mf', 'bp', or 'cc'
ONTOLOGY = 'bp'

# ==============================================================================
# --- SCRIPT LOGIC ---
# ==============================================================================

STATUS_FILE = os.path.join(OUTPUT_DIR, "deepgozero_status.txt")

def update_status(message):
    """Writes a timestamped message to the status file."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(STATUS_FILE, "a") as f:
        f.write(f"{now} - {message}\n")
    print(message)

def validate_interproscan_results(interpro_tsv):
    """Validates that InterProScan results exist and contain data."""
    if not os.path.exists(interpro_tsv):
        raise FileNotFoundError(f"InterProScan results file not found: {interpro_tsv}")
    
    # Count lines in the file
    with open(interpro_tsv, 'r') as f:
        line_count = sum(1 for line in f if line.strip())
    
    if line_count == 0:
        raise ValueError(f"InterProScan results file is empty: {interpro_tsv}")
    
    update_status(f"InterProScan results validated: {line_count} result lines found")
    return line_count

def create_prediction_dataframe(fasta_file, interpro_tsv, output_pkl):
    """Parses InterProScan results and creates the required .pkl file for DeepGOZero."""
    update_status("Creating prediction-ready DataFrame from InterProScan results...")
    
    # Parse InterProScan results to extract InterPro domains
    interpro_map = {}
    total_lines = 0
    interpro_hits = 0
    
    update_status("Parsing InterProScan TSV file...")
    with open(interpro_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 12:  # InterProScan TSV should have at least 12 columns
                continue
                
            total_lines += 1
            protein_id = parts[0]
            
            if protein_id not in interpro_map:
                interpro_map[protein_id] = set()
            
            # Column 11 (index 11) contains InterPro accession
            if len(parts) > 11 and parts[11] and parts[11].startswith("IPR"):
                interpro_map[protein_id].add(parts[11])
                interpro_hits += 1
    
    update_status(f"Parsed {total_lines} InterProScan lines, found {interpro_hits} InterPro domain hits")
    update_status(f"InterPro domains found for {len(interpro_map)} proteins")
    
    # Load protein IDs from original FASTA file to maintain order
    update_status("Loading protein IDs from original FASTA file...")
    protein_ids = []
    fasta_count = 0
    
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            protein_ids.append(record.id)
            fasta_count += 1
    except Exception as e:
        raise ValueError(f"Error reading FASTA file {fasta_file}: {str(e)}")
    
    update_status(f"Loaded {fasta_count} protein IDs from FASTA file")
    
    # Create InterPro lists for each protein
    interpros_list = []
    proteins_with_domains = 0
    proteins_without_domains = 0
    
    for pid in protein_ids:
        interpro_domains = list(interpro_map.get(pid, set()))
        interpros_list.append(interpro_domains)
        
        if interpro_domains:
            proteins_with_domains += 1
        else:
            proteins_without_domains += 1
    
    update_status(f"Proteins with InterPro domains: {proteins_with_domains}")
    update_status(f"Proteins without InterPro domains: {proteins_without_domains}")
    
    # Create DataFrame in the format expected by DeepGOZero
    df = pd.DataFrame({
        'proteins': protein_ids,
        'interpros': interpros_list
    })
    
    # Save as pickle file
    df.to_pickle(output_pkl)
    update_status(f"Created prediction DataFrame with {len(df)} proteins and saved to {output_pkl}")
    
    # Print some statistics
    total_domains = sum(len(interpros) for interpros in interpros_list)
    avg_domains = total_domains / len(protein_ids) if protein_ids else 0
    update_status(f"Average InterPro domains per protein: {avg_domains:.2f}")
    
    return len(df)

def validate_deepgozero_setup(ontology):
    """Validates that DeepGOZero files and data are available."""
    update_status("Validating DeepGOZero setup...")
    
    # Check if DeepGOZero directory exists
    if not os.path.exists(DEEPGOZERO_PATH):
        raise FileNotFoundError(f"DeepGOZero directory not found: {DEEPGOZERO_PATH}")
    
    # Check prediction script
    prediction_script = os.path.join(DEEPGOZERO_PATH, "deepgozero_predict.py")
    if not os.path.exists(prediction_script):
        raise FileNotFoundError(f"DeepGOZero prediction script not found: {prediction_script}")
    
    # Check ontology data directory
    ontology_data_path = os.path.join(DEEPGOZERO_DATA_ROOT, ontology)
    if not os.path.exists(ontology_data_path):
        raise FileNotFoundError(f"DeepGOZero data directory for ontology '{ontology}' not found: {ontology_data_path}")
    
    # Check required model and terms files
    model_file = os.path.join(ontology_data_path, "deepgozero_zero_10.th")
    terms_file = os.path.join(ontology_data_path, "terms_zero_10.pkl")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"DeepGOZero model file not found: {model_file}")
    
    if not os.path.exists(terms_file):
        raise FileNotFoundError(f"DeepGOZero terms file not found: {terms_file}")
    
    update_status("DeepGOZero setup validation successful")
    return prediction_script, model_file, terms_file

def run_deepgozero_prediction(input_pkl, ontology, output_dir):
    """Runs the DeepGOZero prediction script and saves the results."""
    update_status(f"Running DeepGOZero prediction for ontology: {ontology}")
    
    prediction_script, model_file, terms_file = validate_deepgozero_setup(ontology)
    
    cmd = [
        "python", prediction_script, 
        "--test-data-file", input_pkl,
        "--model-file", model_file, 
        "--terms-file", terms_file, 
        "--device", "cuda:0"
    ]
    
    update_status(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run with a reasonable timeout (30 minutes for most datasets)
        process = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            cwd=DEEPGOZERO_PATH,
            timeout=1800  # 30 minutes
        )
        
        update_status("DeepGOZero prediction completed successfully")
        return parse_and_save_results(process.stdout, ontology, input_pkl, output_dir)
        
    except subprocess.TimeoutExpired:
        error_message = "DeepGOZero prediction timed out after 30 minutes"
        update_status(f"ERROR: {error_message}")
        raise RuntimeError(error_message)
    except subprocess.CalledProcessError as e:
        error_message = f"DeepGOZero failed with exit code {e.returncode}. STDERR: {e.stderr}"
        update_status(f"ERROR: {error_message}")
        raise e

def parse_and_save_results(stdout, ontology, input_pkl, output_dir):
    """Parses the prediction output from stdout and saves it to CSV files with analysis."""
    update_status("Parsing and saving final predictions...")
    
    predictions = []
    skipped_lines = 0
    
    for line_num, line in enumerate(stdout.strip().split('\n'), 1):
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
            except (ValueError, IndexError) as e:
                skipped_lines += 1
                update_status(f"Skipped malformed line {line_num}: {line} (Error: {e})")
        else:
            skipped_lines += 1
    
    if skipped_lines > 0:
        update_status(f"Skipped {skipped_lines} malformed output lines")
    
    if not predictions:
        update_status("WARNING: No valid predictions found in DeepGOZero output")
        return 0
    
    # Create DataFrame and save results
    df = pd.DataFrame(predictions)
    
    # Save main results
    csv_path = os.path.join(output_dir, f"predictions_{ontology}.csv")
    df.to_csv(csv_path, index=False)
    update_status(f"Saved {len(df)} predictions to {csv_path}")
    
    # Create and save summary statistics
    summary_path = os.path.join(output_dir, f"prediction_summary_{ontology}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"DEEPGOZERO PREDICTION SUMMARY - {ontology.upper()} ONTOLOGY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total predictions: {len(df)}\n")
        f.write(f"Unique proteins with predictions: {df['Protein_ID'].nunique()}\n")
        f.write(f"Unique GO terms predicted: {df['GO_Term'].nunique()}\n\n")
        
        f.write("Score distribution:\n")
        f.write(f"  Minimum score: {df['Score'].min():.4f}\n")
        f.write(f"  Maximum score: {df['Score'].max():.4f}\n")
        f.write(f"  Mean score: {df['Score'].mean():.4f}\n")
        f.write(f"  Median score: {df['Score'].median():.4f}\n\n")
        
        # Top proteins by number of predictions
        protein_counts = df['Protein_ID'].value_counts().head(10)
        f.write("Top 10 proteins by number of predictions:\n")
        for protein, count in protein_counts.items():
            f.write(f"  {protein}: {count} predictions\n")
        f.write("\n")
        
        # Top GO terms by frequency
        go_counts = df['GO_Term'].value_counts().head(10)
        f.write("Top 10 most frequently predicted GO terms:\n")
        for go_term, count in go_counts.items():
            f.write(f"  {go_term}: {count} proteins\n")
        f.write("\n")
        
        # High confidence predictions (score > 0.5)
        high_conf = df[df['Score'] > 0.5]
        f.write(f"High confidence predictions (score > 0.5): {len(high_conf)}\n")
        f.write(f"Percentage of high confidence: {(len(high_conf)/len(df))*100:.1f}%\n")
    
    update_status(f"Saved prediction summary to {summary_path}")
    
    # Save high-confidence predictions separately
    if len(high_conf) > 0:
        high_conf_path = os.path.join(output_dir, f"high_confidence_predictions_{ontology}.csv")
        high_conf.to_csv(high_conf_path, index=False)
        update_status(f"Saved {len(high_conf)} high-confidence predictions to {high_conf_path}")
    
    return len(df)

@ck.command()
@ck.option('--ontology', default=ONTOLOGY, type=ck.Choice(['mf', 'bp', 'cc']), 
           help="Gene Ontology to predict: 'mf' (molecular function), 'bp' (biological process), 'cc' (cellular component)")
@ck.option('--force-rerun', is_flag=True, help="Force re-running prediction even if output files exist.")
def main(ontology, force_rerun):
    """Main function to run DeepGOZero predictions using InterProScan results."""
    
    # Clear status file
    with open(STATUS_FILE, "w") as f:
        f.write("")
    
    update_status("DeepGOZero prediction pipeline starting...")
    update_status(f"Ontology: {ontology}")
    
    # Define file paths
    interpro_output_tsv = os.path.join(OUTPUT_DIR, "interproscan_results.tsv")
    prediction_input_pkl = os.path.join(OUTPUT_DIR, f"prediction_input_{ontology}.pkl")
    prediction_output_csv = os.path.join(OUTPUT_DIR, f"predictions_{ontology}.csv")
    
    # Check if results already exist and we're not forcing rerun
    if not force_rerun and os.path.exists(prediction_output_csv):
        update_