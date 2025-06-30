import sys
import os
from collections import defaultdict

def combine_prediction_files(bp_file, cc_file, mf_file, output_file):
    """
    Combine three TransFun prediction files (BP, CC, MF) into one file.
    
    Args:
        bp_file: Path to BP predictions
        cc_file: Path to CC predictions  
        mf_file: Path to MF predictions
        output_file: Path to combined output file
    """
    
    # Dictionary to store all predictions per protein
    all_predictions = defaultdict(list)
    
    files_to_process = [
        (bp_file, "BP"),
        (cc_file, "CC"), 
        (mf_file, "MF")
    ]
    
    print("Combining TransFun prediction files...")
    
    for file_path, ontology in files_to_process:
        print(f"Processing {ontology} file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        with open(file_path, 'r') as f:
            for line in f:
                # Skip header lines or metadata
                if line.startswith(("AUTHOR", "MODEL", "KEYWORDS", "END")):
                    continue
                    
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                    
                try:
                    protein_id = parts[0]
                    go_term = parts[1] 
                    score = float(parts[2])
                    
                    # Add to combined predictions
                    all_predictions[protein_id].append((go_term, score))
                    
                except (ValueError, IndexError):
                    continue
    
    # Write combined predictions
    print(f"Writing combined predictions to: {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for protein_id, predictions in all_predictions.items():
            for go_term, score in predictions:
                f.write(f"{protein_id} {go_term} {score}\n")
    
    print(f"Combined predictions complete!")
    print(f"Total proteins: {len(all_predictions)}")
    print(f"Total predictions: {sum(len(preds) for preds in all_predictions.values())}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python combine_predictions.py <bp_file> <cc_file> <mf_file> <output_file>")
        print("Example: python combine_predictions.py transfun_bp.txt transfun_cc.txt transfun_mf.txt combined_transfun.txt")
        sys.exit(1)
    
    bp_file = sys.argv[1]
    cc_file = sys.argv[2]
    mf_file = sys.argv[3]
    output_file = sys.argv[4]
    
    combine_prediction_files(bp_file, cc_file, mf_file, output_file)

if __name__ == "__main__":
    main()