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

def run_evaluation_for_category(category_name, go_tree, truth_data, pred_data, output_file):
    """Runs the full evaluation for a single GO category (BP, CC, or MF)."""

    print(f"\n--- Evaluating Category: {category_name} ---")
    
    # Map from full name to the abbreviation used in the OBO file
    namespace_map = {
        "BP": "biological_process",
        "CC": "cellular_component",
        "MF": "molecular_function"
    }
    namespace = namespace_map[category_name]

    results = []
    for i in range(101):
        threshold = i / 100.0
        
        sum_precision = 0.0
        sum_recall = 0.0
        protein_count = 0
        
        for protein_id, all_true_terms in truth_data.items():
            # Filter the true and predicted terms to only include the current category
            true_set = {go for go in all_true_terms if go_tree.GetGONameSpace(go) == namespace}
            
            # If there are no true terms in this category for this protein, skip it
            if not true_set:
                continue

            protein_count += 1
            predicted_set = set()
            if protein_id in pred_data:
                predicted_set = {go for go, score in pred_data[protein_id] 
                                 if score >= threshold and go_tree.GetGONameSpace(go) == namespace}
            
            precision, recall = go_tree.GOSetsPropagate(predicted_set, true_set)
            
            sum_precision += precision
            sum_recall += recall

        avg_precision = sum_precision / protein_count if protein_count > 0 else 0.0
        avg_recall = sum_recall / protein_count if protein_count > 0 else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

        results.append((threshold, avg_precision, avg_recall, f1_score))
        if i % 10 == 0: # Print progress
             print(f"Threshold: {threshold:.2f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1-Score: {f1_score:.4f}")

    # Save results for this category
    with open(output_file, 'w') as fout:
        fout.write("Threshold\tPropagated_Precision\tPropagated_Recall\tF1-Score\n")
        for res in results:
            fout.write(f"{res[0]:.2f}\t{res[1]:.4f}\t{res[2]:.4f}\t{res[3]:.4f}\n")
    print(f"Results for {category_name} saved to: {output_file}")


def main():
    """
    Main function to run the semantic evaluation using term propagation.
    """
    # --- Hardcoded file paths ---
    OBO_FILE_PATH = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo" 
    
    GROUND_TRUTH_FILE = "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
    PREDICTIONS_FILE = "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt"
    OUTPUT_DIR = "/data/summer2020/Boen/hifun_predictions"

    print("--- Starting Evaluation with Term Propagation (By Category) ---")
    
    # 1. Initialize the Gene Ontology Tree
    go_tree = GeneOntologyTree(OBO_FILE_PATH, TestMode=0)

    # 2. Load all data once
    truth_data = load_ground_truth(GROUND_TRUTH_FILE)
    pred_data = load_predictions(PREDICTIONS_FILE)

    # 3. Run evaluation for each category
    for category in ["BP", "CC", "MF"]:
        output_file_for_category = os.path.join(OUTPUT_DIR, f"hifun_propagation_eval_{category}.tsv")
        run_evaluation_for_category(category, go_tree, truth_data, pred_data, output_file_for_category)
        
    print("\n--- All evaluations complete. ---")

if __name__ == "__main__":
    main()