import sys
import os
from GeneOntologyTree import GeneOntologyTree

class TrueProteinFunction:
    """A class to load ground truth data from a single consolidated file."""
    def __init__(self, ground_truth_file):
        self.AllTrueGO = {}
        self._loadFile(ground_truth_file)

    def _loadFile(self, ground_truth_file):
        print(f"Loading ground truth from: {ground_truth_file}")
        try:
            with open(ground_truth_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2 or parts[1] == 'NOT_FOUND': continue
                    protein_id, go_terms_str = parts[0], parts[1]
                    self.AllTrueGO[protein_id] = set(go_terms_str.split(';'))
        except FileNotFoundError:
            print(f"FATAL ERROR: Ground truth file not found at {ground_truth_file}")
            sys.exit(1)
        print(f"Loaded ground truth for {len(self.AllTrueGO)} proteins.")

    def GetGroundTruth(self):
        return self.AllTrueGO

class PredictedProteinFunction:
    """A class to load prediction data from a single consolidated file."""
    def __init__(self, predictions_file):
        self.AllPredictedGO = {}
        self._loadFile(predictions_file)

    def _loadFile(self, predictions_file):
        print(f"Loading predictions from: {predictions_file}")
        try:
            with open(predictions_file, 'r') as f:
                for line in f:
                    if line.startswith(("AUTHOR", "MODEL", "KEYWORDS", "END")): continue
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    protein_id, go_term, score = parts[0], parts[1], float(parts[2])
                    if protein_id not in self.AllPredictedGO:
                        self.AllPredictedGO[protein_id] = []
                    self.AllPredictedGO[protein_id].append((go_term, score))
        except FileNotFoundError:
            print(f"FATAL ERROR: Predictions file not found at {predictions_file}")
            sys.exit(1)
        print(f"Loaded predictions for {len(self.AllPredictedGO)} proteins.")

def main():
    """Main evaluation function structured to produce a single, combined result file."""
    OBO_FILE_PATH = "/data/shared/go/gene_ontology.obo"
    GROUND_TRUTH_FILE = "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
    PREDICTIONS_FILE = "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt"
    OUTPUT_DIR = "/data/summer2020/Boen/final_evaluation_results"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_results.tsv")

    print("--- Starting Final Combined Evaluation (Corrected Loop) ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Initialize the Gene Ontology Tree
    go_tree = GeneOntologyTree(OBO_FILE_PATH, TestMode=0)

    # 2. Load Ground Truth and Predictions
    TrueGO = TrueProteinFunction(GROUND_TRUTH_FILE)
    PredictedGO = PredictedProteinFunction(PREDICTIONS_FILE)
    
    ListedTrue_ALL = TrueGO.GetGroundTruth()
    AllPredictions = PredictedGO.AllPredictedGO
    
    results = []

    # 4. Main evaluation loop
    print("\nStarting threshold-based evaluation...")
    for i in range(101):
        thres = i / 100.0
        
        total_precision = 0.0
        total_recall = 0.0
        protein_count = 0
        
        # --- LOOP FIX ---
        # Iterate over every protein in the GROUND TRUTH set.
        for protein_id, true_terms in ListedTrue_ALL.items():
            
            # This protein will be included in our evaluation
            protein_count += 1
            
            # Get the predictions for this protein, defaulting to an empty list if none exist
            pred_tuples = AllPredictions.get(protein_id, [])
            
            # Filter predictions based on the current threshold
            predicted_terms = {go for go, score in pred_tuples if score >= thres}

            # Calculate precision and recall using the propagation method
            precision, recall = go_tree.GOSetsPropagate(predicted_terms, true_terms)
            
            total_precision += precision
            total_recall += recall
            
        # Calculate Averages for this threshold
        avg_precision = total_precision / protein_count if protein_count > 0 else 0.0
        avg_recall = total_recall / protein_count if protein_count > 0 else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        results.append((thres, avg_precision, avg_recall, f1_score))

        if i % 10 == 0:
            print(f"Processed Threshold: {thres:.2f} | Avg Precision: {avg_precision:.4f} | Avg Recall: {avg_recall:.4f}")

    # 5. Write the final results to a single file
    print(f"\nWriting final result file to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Threshold\tPrecision\tRecall\tF1-Score\n")
        for res in results:
            f.write(f"{res[0]:.2f}\t{res[1]:.4f}\t{res[2]:.4f}\t{res[3]:.4f}\n")

    print(f"Evaluation complete. Results saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()