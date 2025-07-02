#!/usr/bin/env python3
"""
Convert GAT-GO results to standard format: protein_id \t go_term \t confidence_score
"""

import torch
import numpy as np
import argparse

def load_go_terms(go_map_file=None):
    """
    Load GO term mappings from GAT-GO's go2index.pt file
    """
    if go_map_file is None:
        go_map_file = './data/data_splits/go2index.pt'
    
    try:
        # Load the go2index mapping from GAT-GO
        go2index = torch.load(go_map_file)
        # Reverse the mapping: index -> GO term
        index2go = {v: k for k, v in go2index.items()}
        print(f"Loaded {len(index2go)} GO terms from {go_map_file}")
        return index2go
    except FileNotFoundError:
        print(f"Warning: GO mapping file {go_map_file} not found.")
        print("Please download the GAT-GO data or provide the correct path to go2index.pt")
        # Create dummy GO terms as fallback
        print("Using dummy GO terms...")
        go_terms = {i: f"GO:{i:07d}" for i in range(2752)}
        return go_terms

def convert_results(results_file, output_file, go_map_file=None, threshold=0.5, top_k=None):
    """
    Convert GAT-GO results to standard format
    """
    # Load results
    print(f"Loading results from {results_file}")
    results = torch.load(results_file)
    
    # Load GO terms mapping
    index2go = load_go_terms(go_map_file)
    
    print(f"Converting {len(results)} protein predictions...")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("protein_id\tgo_term\tconfidence_score\n")
        
        for protein_id, predictions in results.items():
            predictions = np.array(predictions)
            
            if top_k is not None:
                # Get top K predictions
                top_indices = np.argsort(predictions)[-top_k:][::-1]
                for idx in top_indices:
                    score = predictions[idx]
                    go_term = index2go.get(idx, f"GO:{idx:07d}")  # Fallback if mapping missing
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")
            else:
                # Use threshold
                high_confidence_indices = np.where(predictions >= threshold)[0]
                for idx in high_confidence_indices:
                    score = predictions[idx]
                    go_term = index2go.get(idx, f"GO:{idx:07d}")  # Fallback if mapping missing
                    f.write(f"{protein_id}\t{go_term}\t{score:.6f}\n")
    
    print(f"Results saved to {output_file}")

def analyze_results(results_file):
    """
    Analyze the results to help choose threshold/top_k
    """
    results = torch.load(results_file)
    
    all_predictions = []
    for predictions in results.values():
        all_predictions.extend(predictions)
    
    all_predictions = np.array(all_predictions)
    
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Min score: {all_predictions.min():.6f}")
    print(f"Max score: {all_predictions.max():.6f}")
    print(f"Mean score: {all_predictions.mean():.6f}")
    print(f"Median score: {np.median(all_predictions):.6f}")
    print(f"95th percentile: {np.percentile(all_predictions, 95):.6f}")
    print(f"99th percentile: {np.percentile(all_predictions, 99):.6f}")
    
    # Count predictions above different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        count = np.sum(all_predictions >= thresh)
        print(f"Predictions >= {thresh}: {count} ({count/len(all_predictions)*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert GAT-GO results to standard format')
    parser.add_argument('--results', default='./results/GAT-GO_Results.pt', help='GAT-GO results file')
    parser.add_argument('--output', default='gat_go_predictions.tsv', help='Output file')
    parser.add_argument('--go_map', default='./data/data_splits/go2index.pt', help='GO terms mapping file (go2index.pt)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--top_k', type=int, help='Take top K predictions per protein (instead of threshold)')
    parser.add_argument('--analyze', action='store_true', help='Analyze results to help choose parameters')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.results)
    else:
        convert_results(args.results, args.output, args.go_map, args.threshold, args.top_k)