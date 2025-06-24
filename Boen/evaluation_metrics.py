#!/usr/bin/env python3
"""
HiFun Protein Function Prediction Evaluator
Evaluates precision and recall of HiFun predictions against ground truth GO terms
Author: Modified from Dr. Renzhi Cao's original script
"""

import sys
import os
from collections import defaultdict

class GeneOntologyTree:
    """Gene Ontology tree class for GO term similarity calculations"""
    
    def __init__(self, obo_path):
        self.GOParent = {}
        self.GOSpace = {}  # store GO id and namespace
        self.MFroot = "GO:0003674"
        self.BProot = "GO:0008150"
        self.CCroot = "GO:0005575"
        print(f"Loading GO tree from {obo_path}")
        self._load_tree(obo_path)
        print(f"Loaded {len(self.GOParent)} GO terms")

    def _load_tree(self, obo_path):
        """Load the OBO file and build the GO tree structure"""
        current_go = None
        parents = []
        namespace = None
        
        with open(obo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(': ', 1)
                if len(parts) < 2:
                    continue
                    
                key, value = parts[0], parts[1]
                
                if key == "id" and value.startswith("GO:"):
                    # Save previous GO term if exists
                    if current_go and current_go != "NULL":
                        self.GOParent[current_go] = parents
                        if namespace:
                            self.GOSpace[current_go] = namespace
                    
                    # Start new GO term
                    current_go = value
                    parents = []
                    namespace = None
                    
                elif key == "is_a" and current_go:
                    parent_go = value.split()[0]  # Take only GO ID, ignore description
                    parents.append(parent_go)
                    
                elif key == "namespace" and current_go:
                    namespace = value
                    
                elif key == "is_obsolete" and current_go:
                    current_go = "NULL"  # Skip obsolete terms
                    
        # Add the last GO term
        if current_go and current_go != "NULL":
            self.GOParent[current_go] = parents
            if namespace:
                self.GOSpace[current_go] = namespace

    def propagate_go_terms(self, go_terms):
        """Propagate GO terms to root, returning all ancestors"""
        propagated = set()
        
        for go_term in go_terms:
            if go_term not in self.GOParent:
                continue
                
            # BFS to find all ancestors
            queue = [go_term]
            visited = set()
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                    
                visited.add(current)
                
                # Don't include root terms in propagated set
                if current not in [self.MFroot, self.BProot, self.CCroot]:
                    propagated.add(current)
                
                # Add parents to queue
                if current in self.GOParent:
                    for parent in self.GOParent[current]:
                        if parent not in visited:
                            queue.append(parent)
        
        return propagated

    def calculate_precision_recall(self, predicted_terms, true_terms):
        """Calculate precision and recall with GO term propagation"""
        if not predicted_terms or not true_terms:
            return 0.0, 0.0
        
        # Propagate both sets
        prop_predicted = self.propagate_go_terms(predicted_terms)
        prop_true = self.propagate_go_terms(true_terms)
        
        if not prop_predicted or not prop_true:
            return 0.0, 0.0
        
        # Calculate TP, FP, FN
        tp = len(prop_predicted & prop_true)  # True positives
        fp = len(prop_predicted - prop_true)  # False positives
        fn = len(prop_true - prop_predicted)  # False negatives
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall


class HiFunEvaluator:
    """Main evaluator class for HiFun predictions"""
    
    def __init__(self):
        # Hardcoded file paths
        self.prediction_file = "/data/summer2020/Boen/hifun_predictions/predictions_for_eval.txt"
        self.ground_truth_file = "/data/summer2020/Boen/ground_truth_go_terms/consolidated_ground_truth.tsv"
        self.obo_file = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
        
        # Initialize GO tree
        self.go_tree = GeneOntologyTree(self.obo_file)
        
        # Load data
        self.predictions = self._load_predictions()
        self.ground_truth = self._load_ground_truth()
        
    def _load_predictions(self):
        """Load HiFun predictions from file"""
        predictions = defaultdict(list)
        
        print(f"Loading predictions from {self.prediction_file}")
        with open(self.prediction_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('AUTHOR') or line.startswith('MODEL') or line.startswith('KEYWORDS'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 3:
                    protein_id = parts[0]
                    go_term = parts[1]
                    score = float(parts[2])
                    predictions[protein_id].append((go_term, score))
        
        print(f"Loaded predictions for {len(predictions)} proteins")
        return predictions
    
    def _load_ground_truth(self):
        """Load ground truth GO terms from TSV file"""
        ground_truth = {}
        
        print(f"Loading ground truth from {self.ground_truth_file}")
        with open(self.ground_truth_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    protein_id = parts[0]
                    go_terms = parts[1].split(';')
                    ground_truth[protein_id] = [term.strip() for term in go_terms if term.strip()]
        
        print(f"Loaded ground truth for {len(ground_truth)} proteins")
        return ground_truth
    
    def evaluate_threshold(self, threshold=0.0):
        """Evaluate predictions using a score threshold"""
        print(f"\nEvaluating with threshold >= {threshold}")
        
        total_precision = 0.0
        total_recall = 0.0
        evaluated_proteins = 0
        
        for protein_id in self.ground_truth:
            if protein_id not in self.predictions:
                continue
            
            # Filter predictions by threshold
            predicted_terms = [
                go_term for go_term, score in self.predictions[protein_id] 
                if score >= threshold
            ]
            
            true_terms = self.ground_truth[protein_id]
            
            if not predicted_terms or not true_terms:
                continue
            
            # Calculate precision and recall for this protein
            precision, recall = self.go_tree.calculate_precision_recall(predicted_terms, true_terms)
            
            total_precision += precision
            total_recall += recall
            evaluated_proteins += 1
        
        if evaluated_proteins == 0:
            return 0.0, 0.0, 0
        
        avg_precision = total_precision / evaluated_proteins
        avg_recall = total_recall / evaluated_proteins
        
        print(f"Evaluated {evaluated_proteins} proteins")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"F1-Score: {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0:.4f}")
        
        return avg_precision, avg_recall, evaluated_proteins
    
    def evaluate_top_n(self, n=10):
        """Evaluate predictions using top N predictions per protein"""
        print(f"\nEvaluating with top {n} predictions per protein")
        
        total_precision = 0.0
        total_recall = 0.0
        evaluated_proteins = 0
        
        for protein_id in self.ground_truth:
            if protein_id not in self.predictions:
                continue
            
            # Sort predictions by score and take top N
            sorted_predictions = sorted(self.predictions[protein_id], key=lambda x: x[1], reverse=True)
            predicted_terms = [go_term for go_term, _ in sorted_predictions[:n]]
            
            true_terms = self.ground_truth[protein_id]
            
            if not predicted_terms or not true_terms:
                continue
            
            # Calculate precision and recall for this protein
            precision, recall = self.go_tree.calculate_precision_recall(predicted_terms, true_terms)
            
            total_precision += precision
            total_recall += recall
            evaluated_proteins += 1
        
        if evaluated_proteins == 0:
            return 0.0, 0.0, 0
        
        avg_precision = total_precision / evaluated_proteins
        avg_recall = total_recall / evaluated_proteins
        
        print(f"Evaluated {evaluated_proteins} proteins")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"F1-Score: {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0:.4f}")
        
        return avg_precision, avg_recall, evaluated_proteins
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with multiple thresholds and top-N values"""
        print("="*60)
        print("HiFun Protein Function Prediction Evaluation")
        print("="*60)
        
        # Threshold-based evaluation
        print("\n" + "="*40)
        print("THRESHOLD-BASED EVALUATION")
        print("="*40)
        
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            self.evaluate_threshold(threshold)
        
        # Top-N based evaluation
        print("\n" + "="*40)
        print("TOP-N BASED EVALUATION")
        print("="*40)
        
        top_n_values = [1, 3, 5, 10, 15, 20]
        for n in top_n_values:
            self.evaluate_top_n(n)
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)


def main():
    """Main function"""
    try:
        evaluator = HiFunEvaluator()
        evaluator.run_comprehensive_evaluation()
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()