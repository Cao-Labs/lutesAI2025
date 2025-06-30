import sys
import os
import csv
import re
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
                    if len(parts) < 2 or parts[1] == 'NOT_FOUND': 
                        continue
                    protein_id, go_terms_str = parts[0], parts[1]
                    self.AllTrueGO[protein_id] = set(go_terms_str.split(';'))
        except FileNotFoundError:
            print(f"FATAL ERROR: Ground truth file not found at {ground_truth_file}")
            sys.exit(1)
        print(f"Loaded ground truth for {len(self.AllTrueGO)} proteins.")

    def GetGroundTruth(self):
        return self.AllTrueGO

class PredictedProteinFunction:
    """A class to load prediction data from various formats (CSV, TSV, TXT, space-separated)."""
    def __init__(self, predictions_file):
        self.AllPredictedGO = {}
        self._loadFile(predictions_file)

    def _loadFile(self, predictions_file):
        print(f"Loading predictions from: {predictions_file}")
        try:
            # Try different loading methods in order of preference
            if predictions_file.lower().endswith('.csv'):
                self._loadCSV(predictions_file)
            else:
                # For .txt, .tsv, or any other format, try flexible parsing
                self._loadFlexible(predictions_file)
        except FileNotFoundError:
            print(f"FATAL ERROR: Predictions file not found at {predictions_file}")
            sys.exit(1)
        print(f"Loaded predictions for {len(self.AllPredictedGO)} proteins.")

    def _detect_delimiter(self, sample_lines):
        """Detect the most likely delimiter from sample lines."""
        delimiters = ['\t', ' ', ',', ';', '|']
        delimiter_counts = {}
        
        for delimiter in delimiters:
            count = 0
            for line in sample_lines:
                if line.strip():
                    parts = line.strip().split(delimiter)
                    if len(parts) >= 3:  # We need at least 3 columns
                        try:
                            # Check if the third column can be converted to float
                            float(parts[2])
                            count += 1
                        except (ValueError, IndexError):
                            pass
            delimiter_counts[delimiter] = count
        
        # Return the delimiter with the highest success count
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        print(f"Auto-detected delimiter: '{best_delimiter}' (found {delimiter_counts[best_delimiter]} valid lines)")
        return best_delimiter

    def _loadFlexible(self, predictions_file):
        """Load predictions with automatic delimiter detection."""
        with open(predictions_file, 'r') as f:
            # Read first few lines to detect delimiter
            sample_lines = []
            file_position = f.tell()
            for _ in range(10):  # Sample first 10 lines
                line = f.readline()
                if not line:
                    break
                # Skip common header patterns
                if not line.startswith(("AUTHOR", "MODEL", "KEYWORDS", "END")):
                    sample_lines.append(line)
            
            # Reset file position
            f.seek(file_position)
            
            if not sample_lines:
                print("WARNING: No valid sample lines found for delimiter detection")
                return
            
            # Detect delimiter
            delimiter = self._detect_delimiter(sample_lines)
            
            # Process the entire file
            f.seek(0)
            for line in f:
                if line.startswith(("AUTHOR", "MODEL", "KEYWORDS", "END")): 
                    continue
                
                # Handle different delimiters
                if delimiter == ' ':
                    # For space-separated, split on whitespace and handle multiple spaces
                    parts = line.strip().split()
                else:
                    parts = line.strip().split(delimiter)
                
                if len(parts) < 3: 
                    continue
                
                try:
                    protein_id, go_term, score = parts[0], parts[1], float(parts[2])
                    if protein_id not in self.AllPredictedGO:
                        self.AllPredictedGO[protein_id] = []
                    self.AllPredictedGO[protein_id].append((go_term, score))
                except (ValueError, IndexError):
                    continue

    def _loadCSV(self, predictions_file):
        """Load predictions from CSV file with automatic delimiter detection."""
        with open(predictions_file, 'r') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            try:
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                print(f"CSV delimiter detected: '{delimiter}'")
            except:
                # Fallback to comma if detection fails
                delimiter = ','
                print("CSV delimiter detection failed, using comma as default")
            
            reader = csv.reader(f, delimiter=delimiter)
            
            # Skip header if it exists
            try:
                first_row = next(reader)
                if not self._is_data_row(first_row):
                    pass  # Header was skipped
                else:
                    # First row is data, process it
                    self._process_prediction_row(first_row)
            except StopIteration:
                return
            
            # Process remaining rows
            for row in reader:
                self._process_prediction_row(row)

    def _is_data_row(self, row):
        """Check if a row contains data (not a header)."""
        if len(row) < 3:
            return False
        try:
            float(row[2])  # Try to convert score to float
            return True
        except (ValueError, IndexError):
            return False

    def _process_prediction_row(self, row):
        """Process a single prediction row."""
        if len(row) < 3:
            return
        try:
            protein_id, go_term, score = row[0].strip(), row[1].strip(), float(row[2])
            if protein_id not in self.AllPredictedGO:
                self.AllPredictedGO[protein_id] = []
            self.AllPredictedGO[protein_id].append((go_term, score))
        except (ValueError, IndexError):
            pass

def evaluate_predictions(obo_file, ground_truth_file, predictions_file, output_dir):
    """
    Main evaluation function.
    
    Args:
        obo_file: Path to Gene Ontology OBO file
        ground_truth_file: Path to ground truth file (TSV format)
        predictions_file: Path to predictions file (supports CSV, TSV, TXT, space-separated)
        output_dir: Directory to save results
    """
    
    output_file = os.path.join(output_dir, "evaluation_results.tsv")
    
    print("--- Starting Protein Function Evaluation ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Initialize the Gene Ontology Tree
    print("Loading Gene Ontology tree...")
    go_tree = GeneOntologyTree(obo_file, TestMode=0)

    # 2. Load Ground Truth and Predictions
    print("Loading ground truth and predictions...")
    TrueGO = TrueProteinFunction(ground_truth_file)
    PredictedGO = PredictedProteinFunction(predictions_file)
    
    ListedTrue_ALL = TrueGO.GetGroundTruth()
    AllPredictions = PredictedGO.AllPredictedGO
    
    print(f"Ground truth proteins: {len(ListedTrue_ALL)}")
    print(f"Prediction proteins: {len(AllPredictions)}")
    
    # Find overlap
    common_proteins = set(ListedTrue_ALL.keys()) & set(AllPredictions.keys())
    print(f"Common proteins for evaluation: {len(common_proteins)}")
    
    if len(common_proteins) == 0:
        print("WARNING: No common proteins found between ground truth and predictions!")
        print("Sample ground truth protein IDs:", list(ListedTrue_ALL.keys())[:5])
        print("Sample prediction protein IDs:", list(AllPredictions.keys())[:5])
        return
    
    results = []

    # 3. Main evaluation loop
    print("\nStarting threshold-based evaluation...")
    for i in range(101):
        thres = i / 100.0
        
        total_precision = 0.0
        total_recall = 0.0
        protein_count = 0
        
        # Iterate over proteins that have both ground truth and predictions
        for protein_id in common_proteins:
            true_terms = ListedTrue_ALL[protein_id]
            pred_tuples = AllPredictions[protein_id]
            
            protein_count += 1
            
            # Filter predictions based on the current threshold
            predicted_terms = {go for go, score in pred_tuples if score >= thres}

            # Calculate precision and recall using the propagation method
            precision, recall = go_tree.GOSetsPropagate(predicted_terms, true_terms)
            
            total_precision += precision
            total_recall += recall
            
        # Calculate averages for this threshold
        avg_precision = total_precision / protein_count if protein_count > 0 else 0.0
        avg_recall = total_recall / protein_count if protein_count > 0 else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        results.append((thres, avg_precision, avg_recall, f1_score))

        if i % 10 == 0:
            print(f"Threshold: {thres:.2f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {f1_score:.4f}")

    # 4. Write results
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("Threshold\tPrecision\tRecall\tF1-Score\n")
        for res in results:
            f.write(f"{res[0]:.2f}\t{res[1]:.4f}\t{res[2]:.4f}\t{res[3]:.4f}\n")

    # 5. Find best F1 score
    best_result = max(results, key=lambda x: x[3])
    print(f"\nBest F1 Score: {best_result[3]:.4f} at threshold {best_result[0]:.2f}")
    print(f"  Precision: {best_result[1]:.4f}")
    print(f"  Recall: {best_result[2]:.4f}")
    
    print(f"\nEvaluation complete! Results saved in {output_file}")

def main():
    """Main function with configurable file paths."""
    
    # Default file paths - modify these for your setup
    OBO_FILE_PATH = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
    GROUND_TRUTH_FILE = "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
    PREDICTIONS_FILE = "/data/summer2020/Boen/transfun_predictions/transfun_predictions_bp.txt"  # Updated for TransFun
    OUTPUT_DIR = "/data/summer2020/Boen/transfun_evaluation_output"
    
    # Check if command line arguments are provided
    if len(sys.argv) >= 4:
        OBO_FILE_PATH = sys.argv[1]
        GROUND_TRUTH_FILE = sys.argv[2]
        PREDICTIONS_FILE = sys.argv[3]
        if len(sys.argv) >= 5:
            OUTPUT_DIR = sys.argv[4]
    
    evaluate_predictions(OBO_FILE_PATH, GROUND_TRUTH_FILE, PREDICTIONS_FILE, OUTPUT_DIR)

if __name__ == "__main__":
    main()