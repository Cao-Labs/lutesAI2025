#!/usr/bin/env python3
"""
DeepGOZero Evaluation Pipeline
Converts DeepGOZero predictions to evaluation format and runs the evaluation script.
"""

import os
import sys
import pandas as pd
import subprocess
from datetime import datetime

# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS TO MATCH YOUR SYSTEM
# ==============================================================================

# Path to DeepGOZero output directory (from the prediction script)
DEEPGOZERO_OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_pipeline_output"

# Path to your evaluation script (the first script you provided)
EVALUATION_SCRIPT = "/data/shared/github/lutesAI2025/evaluation_metrics.py"  # UPDATE THIS PATH

# Which ontology was predicted: 'mf', 'bp', or 'cc' 
ONTOLOGY = 'bp'  # Change this to match what you ran DeepGOZero on

# Output directory for evaluation results
EVALUATION_OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_predictions"

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def log_message(message):
    """Print timestamped log messages."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def convert_deepgozero_to_evaluation_format(csv_file, output_file):
    """
    Convert DeepGOZero CSV predictions to the TSV format expected by evaluation script.
    
    DeepGOZero format: Protein_ID,GO_Term,Score
    Evaluation format: Protein_ID\tGO_Term\tScore (with header handling)
    """
    log_message(f"Converting DeepGOZero predictions from {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"DeepGOZero predictions file not found: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    log_message(f"Loaded {len(df)} predictions for {df['Protein_ID'].nunique()} proteins")
    
    # Validate required columns
    required_cols = ['Protein_ID', 'GO_Term', 'Score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DeepGOZero output: {missing_cols}")
    
    # Sort by protein ID and score (descending) for better organization
    df_sorted = df.sort_values(['Protein_ID', 'Score'], ascending=[True, False])
    
    # Write in the format expected by evaluation script
    # The evaluation script expects: protein_id \t go_term \t score
    # And it skips lines starting with AUTHOR, MODEL, KEYWORDS, END
    with open(output_file, 'w') as f:
        # Add some metadata that the evaluation script will skip
        f.write("AUTHOR\tDeepGOZero_Pipeline\n")
        f.write("MODEL\tDeepGOZero\n")
        f.write(f"KEYWORDS\tOntology:{ONTOLOGY}\n")
        
        # Write the actual predictions
        for _, row in df_sorted.iterrows():
            f.write(f"{row['Protein_ID']}\t{row['GO_Term']}\t{row['Score']:.6f}\n")
        
        f.write("END\n")
    
    log_message(f"Converted predictions saved to: {output_file}")
    return len(df_sorted)

def validate_evaluation_setup():
    """Validate that all required files for evaluation exist."""
    log_message("Validating evaluation setup...")
    
    # Check if evaluation script exists
    if not os.path.exists(EVALUATION_SCRIPT):
        raise FileNotFoundError(f"Evaluation script not found: {EVALUATION_SCRIPT}")
    
    # The evaluation script has these hardcoded paths - let's check if they exist
    # You may need to update these paths in the evaluation script
    required_files = [
        "/data/shared/databases/UniProt2025/GO_June_1_2025.obo",
        "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        log_message("WARNING: The following required files for evaluation are missing:")
        for file_path in missing_files:
            log_message(f"  - {file_path}")
        log_message("You may need to update the paths in the evaluation script.")
        return False
    
    log_message("Evaluation setup validation completed successfully")
    return True

def run_evaluation(predictions_file):
    """Run the evaluation script with the converted predictions."""
    log_message("Starting evaluation...")
    
    # Create a temporary modified version of the evaluation script with our prediction file
    temp_eval_script = os.path.join(EVALUATION_OUTPUT_DIR, "temp_evaluation_script.py")
    
    # Read the original evaluation script
    with open(EVALUATION_SCRIPT, 'r') as f:
        eval_code = f.read()
    
    # Replace the predictions file path
    eval_code = eval_code.replace(
        'PREDICTIONS_FILE = "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt"',
        f'PREDICTIONS_FILE = "{predictions_file}"'
    )
    
    # Update output directory
    eval_code = eval_code.replace(
        'OUTPUT_DIR = "/data/summer2020/Boen/final_evaluation_results"',
        f'OUTPUT_DIR = "{EVALUATION_OUTPUT_DIR}"'
    )
    
    # Write temporary script
    with open(temp_eval_script, 'w') as f:
        f.write(eval_code)
    
    # Run the evaluation
    try:
        log_message(f"Running evaluation script: {temp_eval_script}")
        result = subprocess.run([sys.executable, temp_eval_script], 
                              capture_output=True, text=True, check=True)
        
        log_message("Evaluation completed successfully!")
        log_message("STDOUT from evaluation:")
        print(result.stdout)
        
        if result.stderr:
            log_message("STDERR from evaluation:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        log_message(f"ERROR: Evaluation script failed with exit code {e.returncode}")
        log_message("STDOUT:")
        print(e.stdout)
        log_message("STDERR:")
        print(e.stderr)
        raise
    finally:
        # Clean up temporary script
        if os.path.exists(temp_eval_script):
            os.remove(temp_eval_script)

def main():
    """Main function to convert DeepGOZero output and run evaluation."""
    log_message("=== DeepGOZero Evaluation Pipeline Starting ===")
    log_message(f"Ontology: {ONTOLOGY}")
    
    # Create output directory
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
    
    try:
        # Step 1: Find DeepGOZero prediction file
        deepgozero_csv = os.path.join(DEEPGOZERO_OUTPUT_DIR, f"predictions_{ONTOLOGY}.csv")
        if not os.path.exists(deepgozero_csv):
            raise FileNotFoundError(f"DeepGOZero predictions not found: {deepgozero_csv}")
        
        # Step 2: Convert to evaluation format
        converted_predictions = os.path.join(EVALUATION_OUTPUT_DIR, f"deepgozero_predictions_{ONTOLOGY}_for_eval.txt")
        prediction_count = convert_deepgozero_to_evaluation_format(deepgozero_csv, converted_predictions)
        
        # Step 3: Validate evaluation setup
        if not validate_evaluation_setup():
            log_message("WARNING: Some evaluation files may be missing. Proceeding anyway...")
        
        # Step 4: Run evaluation
        run_evaluation(converted_predictions)
        
        # Step 5: Report results
        results_file = os.path.join(EVALUATION_OUTPUT_DIR, "evaluation_results.tsv")
        if os.path.exists(results_file):
            log_message(f"Evaluation results saved to: {results_file}")
            
            # Show a preview of results
            log_message("Preview of evaluation results:")
            with open(results_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:6]):  # Show header + first 5 results
                    print(f"  {line.strip()}")
                if len(lines) > 6:
                    print(f"  ... and {len(lines)-6} more rows")
        
        log_message("=== DeepGOZero Evaluation Pipeline Completed Successfully! ===")
        log_message(f"Evaluated {prediction_count} predictions")
        log_message(f"Results directory: {EVALUATION_OUTPUT_DIR}")
        
    except Exception as e:
        log_message(f"ERROR: Pipeline failed - {str(e)}")
        raise

if __name__ == "__main__":
    # You can also add command line arguments here if needed
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation on DeepGOZero predictions")
    parser.add_argument("--ontology", choices=['mf', 'bp', 'cc'], default=ONTOLOGY,
                       help="Gene Ontology to evaluate")
    parser.add_argument("--deepgozero-dir", default=DEEPGOZERO_OUTPUT_DIR,
                       help="Directory containing DeepGOZero predictions")
    parser.add_argument("--eval-script", default=EVALUATION_SCRIPT,
                       help="Path to evaluation script")
    parser.add_argument("--output-dir", default=EVALUATION_OUTPUT_DIR,
                       help="Directory for evaluation results")
    
    args = parser.parse_args()
    
    # Update global variables with command line arguments
    ONTOLOGY = args.ontology
    DEEPGOZERO_OUTPUT_DIR = args.deepgozero_dir
    EVALUATION_SCRIPT = args.eval_script
    EVALUATION_OUTPUT_DIR = args.output_dir
    
    main()