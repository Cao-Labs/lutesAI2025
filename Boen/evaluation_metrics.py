import sys
import os
from GeneOntologyTree import GeneOntologyTree # Import the professor's class

def load_ground_truth(ground_truth_file):
    """Loads the ground truth file into a dictionary."""
    ground_truth = {}
    print(f"Loading ground truth from: {ground_truth_file}")
    try:
        with open(ground_truth_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2 or parts[1] == 'NOT_FOUND':
                    continue
                protein_id, go_terms_str = parts[0], parts[1]
                ground_truth[protein_id] = set(go_terms_str.split(';'))
    except FileNotFoundError:
        print(f"FATAL ERROR: Ground truth file not found at {ground_truth_file}")
        sys.exit(1)
    print(f"Loaded ground truth for {len(ground_truth)} proteins.")
    return ground_truth

def load_predictions(predictions_file):
    """Loads the model's prediction file into a dictionary."""
    predictions = {}
    print(f"Loading predictions from: {predictions_file}")
    try:
        with open(predictions_file, 'r') as f:
            for line in f:
                if line.startswith("AUTHOR") or line.startswith("MODEL") or \
                   line.startswith("KEYWORDS") or line.startswith("END"):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                protein_id, go_term, score = parts[0], parts[1], float(parts[2])
                if protein_id not in predictions:
                    predictions[protein_id] = []
                predictions[protein_id].append((go_term, score))
    except FileNotFoundError:
        print(f"FATAL ERROR: Predictions file not found at {predictions_file}")
        sys.exit(1)
    print(f"Loaded predictions for {len(predictions)} proteins.")
    return predictions

def main():
    """
    Main function to run the semantic evaluation using term propagation.
    """
    # --- Hardcoded file paths ---
    # !!! IMPORTANT !!!
    # You may need to update this path to the correct location of your gene ontology file.
    OBO_FILE_PATH = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo" 
    
    GROUND_TRUTH_FILE = "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
    PREDICTIONS_FILE = "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt"
    OUTPUT_DIR = "/data/summer2020/Boen/hifun_predictions"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hifun_propagation_evaluation_results.tsv")

    print("--- Starting Evaluation with Term Propagation ---")
    print(f"OBO File: {OBO_FILE_PATH}")
    print(f"Ground Truth: {GROUND_TRUTH_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("------------------------------------------------\n")

    # 1. Initialize the Gene Ontology Tree using the professor's class
    go_tree = GeneOntologyTree(OBO_FILE_PATH, TestMode=0) # Set TestMode=0 to reduce print statements

    # 2. Load the ground truth and prediction data
    truth_data = load_ground_truth(GROUND_TRUTH_FILE)
    pred_data = load_predictions(PREDICTIONS_FILE)

    # 3. Run the evaluation
    print("\nCalculating propagated precision and recall at different thresholds...")
    results = []
    
    for i in range(101):
        threshold = i / 100.0
        
        sum_precision = 0.0
        sum_recall = 0.0
        protein_count = 0
        
        for protein_id, true_set in truth_data.items():
            if protein_id in pred_data:
                protein_count += 1
                
                # Get predicted terms above the current threshold
                predicted_set = {go for go, score in pred_data[protein_id] if score >= threshold}
                
                # Use the professor's propagation method for evaluation
                precision, recall = go_tree.GOSetsPropagate(predicted_set, true_set)
                
                sum_precision += precision
                sum_recall += recall

        avg_precision = sum_precision / protein_count if protein_count > 0 else 0.0
        avg_recall = sum_recall / protein_count if protein_count > 0 else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

        results.append((threshold, avg_precision, avg_recall, f1_score))
        print(f"Threshold: {threshold:.2f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {f1_score:.4f}")

    # 4. Save results to file
    print(f"\nSaving propagation evaluation results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as fout:
        fout.write("Threshold\tPropagated_Precision\tPropagated_Recall\tF1-Score\n")
        for res in results:
            fout.write(f"{res[0]:.2f}\t{res[1]:.4f}\t{res[2]:.4f}\t{res[3]:.4f}\n")
    print("Evaluation complete.")

if __name__ == "__main__":
    main()