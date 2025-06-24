import sys
import os

def load_ground_truth(ground_truth_file):
    """
    Loads the ground truth file into a dictionary.

    Args:
        ground_truth_file (str): Path to the consolidated_ground_truth.tsv file.

    Returns:
        dict: A dictionary mapping a protein ID to a set of its true GO terms.
              Example: {'protein1': {'GO:001', 'GO:002'}, ...}
    """
    ground_truth = {}
    print(f"Loading ground truth from: {ground_truth_file}")
    try:
        with open(ground_truth_file, 'r') as f:
            next(f)  # Skip the header line
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2 or parts[1] == 'NOT_FOUND':
                    continue
                protein_id = parts[0]
                go_terms = set(parts[1].split(';'))
                ground_truth[protein_id] = go_terms
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        sys.exit(1)
    print(f"Loaded ground truth for {len(ground_truth)} proteins.")
    return ground_truth

def load_predictions(predictions_file):
    """
    Loads the model's prediction file into a dictionary.

    Args:
        predictions_file (str): Path to the predictions_for_eval.txt file.

    Returns:
        dict: A dictionary mapping a protein ID to a list of (GO term, score) tuples.
              Example: {'protein1': [('GO:001', 0.95), ('GO:003', 0.8)...], ...}
    """
    predictions = {}
    print(f"Loading predictions from: {predictions_file}")
    try:
        with open(predictions_file, 'r') as f:
            for line in f:
                # Skip header/footer lines from the prediction file format
                if line.startswith("AUTHOR") or line.startswith("MODEL") or \
                   line.startswith("KEYWORDS") or line.startswith("END"):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                protein_id, go_term, score = parts[0], parts[1], float(parts[2])
                
                if protein_id not in predictions:
                    predictions[protein_id] = []
                predictions[protein_id].append((go_term, score))
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_file}")
        sys.exit(1)
    print(f"Loaded predictions for {len(predictions)} proteins.")
    return predictions

def evaluate_performance(ground_truth, predictions, output_file):
    """
    Calculates precision and recall at various thresholds and saves the results.

    Args:
        ground_truth (dict): The dictionary of true GO terms.
        predictions (dict): The dictionary of predicted GO terms with scores.
        output_file (str): Path to save the evaluation results.
    """
    print("\nStarting evaluation...")
    
    results = []
    
    # Ensure the output directory exists
    output_dir_path = os.path.dirname(output_file)
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Iterate through thresholds from 0.00 to 1.00
    for i in range(101):
        threshold = i / 100.0
        
        total_true_positives = 0
        total_predicted_positives = 0
        total_actual_positives = 0
        
        # Evaluate for each protein that has ground truth data
        for protein_id, true_go_terms in ground_truth.items():
            
            # Get the set of true GO terms for this protein
            true_set = true_go_terms
            total_actual_positives += len(true_set)

            # Get predictions for this protein, if they exist
            if protein_id in predictions:
                # Filter predictions based on the current threshold
                predicted_set = {go for go, score in predictions[protein_id] if score >= threshold}
                
                # Calculate metrics for this protein
                true_positives = len(predicted_set.intersection(true_set))
                
                total_true_positives += true_positives
                total_predicted_positives += len(predicted_set)
        
        # Calculate overall precision and recall for this threshold
        precision = total_true_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0
        recall = total_true_positives / total_actual_positives if total_actual_positives > 0 else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append((threshold, precision, recall, f1_score))
        print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    # Save the final results to a file
    print(f"\nSaving evaluation results to: {output_file}")
    with open(output_file, 'w') as fout:
        fout.write("Threshold\tPrecision\tRecall\tF1-Score\n")
        for res in results:
            fout.write(f"{res[0]:.2f}\t{res[1]:.4f}\t{res[2]:.4f}\t{res[3]:.4f}\n")
    print("Evaluation complete.")

def main():
    """
    Main function to run the evaluation pipeline with hardcoded paths.
    """
    # --- Hardcoded file paths ---
    GROUND_TRUTH_FILE = "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
    PREDICTIONS_FILE = "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt"
    OUTPUT_DIR = "/data/summer2020/Boen/hifun_predictions"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hifun_evaluation_results.tsv")
    
    print("--- Starting Precision-Recall Evaluation ---")
    print(f"Ground Truth File: {GROUND_TRUTH_FILE}")
    print(f"Predictions File: {PREDICTIONS_FILE}")
    print(f"Output File: {OUTPUT_FILE}")
    print("------------------------------------------\n")

    # 1. Load the ground truth and prediction data
    truth_data = load_ground_truth(GROUND_TRUTH_FILE)
    pred_data = load_predictions(PREDICTIONS_FILE)

    # 2. Run the evaluation and save the results
    evaluate_performance(truth_data, pred_data, OUTPUT_FILE)


if __name__ == "__main__":
    main()
