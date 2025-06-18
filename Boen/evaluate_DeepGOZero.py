# -*- coding: utf-8 -*-
"""
DeepGOZero Protein Function Prediction Script
Zero-shot GO term predictions using ontology embeddings
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle
import argparse
from collections import defaultdict
import subprocess
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# DeepGOZero repository path
DEEPGOZERO_PATH = "/data/shared/tools/deepgozero"
sys.path.append(DEEPGOZERO_PATH)

# Assuming these imports based on the repository structure
try:
    import torch
    import torch.nn as nn
    from utils import Ontology, FUNC_DICT, NAMESPACES
    from deepgozero_predict import DeepGOZeroModel, load_model_and_data
except ImportError as e:
    logging.warning(f"Could not import DeepGOZero modules: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths - adjust these based on your setup
DEFAULT_FASTA = "/data/summer2020/naufal/testing_sequences.fasta"
DEFAULT_OUTPUT_DIR = "/data/summer2020/Boen/deepgozero_output"
DIAMOND_DB = "/data/shared/tools/deepgozero/data/swissprot_exp.fasta"
GO_OBO_FILE = "/data/shared/tools/deepgozero/data/go.obo"

def run_diamond_search(query_fasta, database, output_file, max_target_seqs=10000):
    """
    Run Diamond BLASTP search for sequence similarity
    
    Args:
        query_fasta (str): Path to query FASTA file
        database (str): Path to Diamond database
        output_file (str): Output file for Diamond results
        max_target_seqs (int): Maximum number of target sequences
    
    Returns:
        str: Path to Diamond results file
    """
    logging.info("Running Diamond BLASTP search...")
    
    # Create Diamond database if it doesn't exist
    db_file = database + ".dmnd"
    if not os.path.exists(db_file):
        logging.info("Creating Diamond database...")
        cmd = f"diamond makedb --in {database} --db {database}"
        subprocess.run(cmd, shell=True, check=True)
    
    # Run Diamond search
    cmd = [
        "diamond", "blastp",
        "--query", query_fasta,
        "--db", database,
        "--out", output_file,
        "--outfmt", "6", "qseqid", "sseqid", "bitscore",
        "--max-target-seqs", str(max_target_seqs),
        "--evalue", "0.001"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Diamond search completed. Results saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Diamond search failed: {e}")
        raise

def load_training_annotations(data_file):
    """
    Load training annotations for similarity-based predictions
    
    Args:
        data_file (str): Path to training data pickle file
    
    Returns:
        dict: Dictionary mapping protein IDs to GO annotations
    """
    logging.info(f"Loading training annotations from {data_file}")
    
    try:
        train_df = pd.read_pickle(data_file)
        annotations = {}
        
        for _, row in train_df.iterrows():
            prot_id = row['proteins']
            annots = set(row['prop_annotations'])
            annotations[prot_id] = annots
            
        logging.info(f"Loaded annotations for {len(annotations)} proteins")
        return annotations
    except Exception as e:
        logging.error(f"Failed to load training annotations: {e}")
        return {}

def predict_with_similarity(diamond_results, annotations, ontology):
    """
    Make predictions based on sequence similarity (Diamond results)
    
    Args:
        diamond_results (str): Path to Diamond results file
        annotations (dict): Training protein annotations
        ontology (Ontology): GO ontology object
    
    Returns:
        dict: Predictions for each query protein
    """
    logging.info("Making similarity-based predictions...")
    
    similarity_preds = defaultdict(dict)
    
    try:
        with open(diamond_results, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id = parts[0]
                    target_id = parts[1]
                    bitscore = float(parts[2])
                    
                    # Get annotations for target protein
                    if target_id in annotations:
                        target_annots = annotations[target_id]
                        
                        # Transfer annotations weighted by similarity score
                        for go_term in target_annots:
                            if go_term in similarity_preds[query_id]:
                                similarity_preds[query_id][go_term] = max(
                                    similarity_preds[query_id][go_term], 
                                    bitscore
                                )
                            else:
                                similarity_preds[query_id][go_term] = bitscore
        
        # Normalize scores
        for query_id in similarity_preds:
            max_score = max(similarity_preds[query_id].values())
            if max_score > 0:
                for go_term in similarity_preds[query_id]:
                    similarity_preds[query_id][go_term] /= max_score
    
    except Exception as e:
        logging.error(f"Error in similarity-based prediction: {e}")
    
    logging.info(f"Generated similarity predictions for {len(similarity_preds)} proteins")
    return dict(similarity_preds)

def predict_with_deepgozero(fasta_file, model_path, terms_file, ontology):
    """
    Make predictions using DeepGOZero model
    
    Args:
        fasta_file (str): Path to input FASTA file
        model_path (str): Path to trained DeepGOZero model
        terms_file (str): Path to GO terms file
        ontology (Ontology): GO ontology object
    
    Returns:
        dict: DeepGOZero predictions for each protein
    """
    logging.info("Making DeepGOZero predictions...")
    
    try:
        # Load model and terms
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        
        # This is a simplified version - actual implementation would depend on
        # the specific DeepGOZero model architecture and loading functions
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Load sequences
        sequences = {}
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq)
        
        deepgozero_preds = defaultdict(dict)
        
        # Process each sequence
        for seq_id, sequence in sequences.items():
            # Convert sequence to model input format
            # This would need to be implemented based on DeepGOZero's input processing
            seq_input = preprocess_sequence(sequence)  # Placeholder function
            
            with torch.no_grad():
                # Get model predictions
                predictions = model(seq_input)
                predictions = torch.sigmoid(predictions).numpy()
                
                # Map predictions to GO terms
                for i, score in enumerate(predictions):
                    if i < len(terms):
                        go_term = terms[i]
                        deepgozero_preds[seq_id][go_term] = float(score)
        
        logging.info(f"Generated DeepGOZero predictions for {len(deepgozero_preds)} proteins")
        return dict(deepgozero_preds)
        
    except Exception as e:
        logging.error(f"Error in DeepGOZero prediction: {e}")
        return {}

def preprocess_sequence(sequence, max_length=1000):
    """
    Preprocess protein sequence for model input
    This is a placeholder - actual implementation depends on DeepGOZero's preprocessing
    
    Args:
        sequence (str): Protein sequence
        max_length (int): Maximum sequence length
    
    Returns:
        torch.Tensor: Preprocessed sequence tensor
    """
    # Amino acid to index mapping
    aa_to_idx = {
        'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8,
        'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
        'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'X': 21
    }
    
    # Convert sequence to indices
    seq_indices = [aa_to_idx.get(aa, 21) for aa in sequence[:max_length]]
    
    # Pad or truncate to max_length
    if len(seq_indices) < max_length:
        seq_indices.extend([0] * (max_length - len(seq_indices)))
    
    return torch.tensor(seq_indices, dtype=torch.long).unsqueeze(0)

def combine_predictions(similarity_preds, deepgozero_preds, alpha=0.5):
    """
    Combine similarity-based and DeepGOZero predictions
    
    Args:
        similarity_preds (dict): Similarity-based predictions
        deepgozero_preds (dict): DeepGOZero predictions
        alpha (float): Weight for similarity predictions (1-alpha for DeepGOZero)
    
    Returns:
        dict: Combined predictions
    """
    logging.info("Combining predictions...")
    
    combined_preds = defaultdict(dict)
    all_proteins = set(similarity_preds.keys()) | set(deepgozero_preds.keys())
    
    for protein_id in all_proteins:
        sim_pred = similarity_preds.get(protein_id, {})
        deep_pred = deepgozero_preds.get(protein_id, {})
        
        all_terms = set(sim_pred.keys()) | set(deep_pred.keys())
        
        for go_term in all_terms:
            sim_score = sim_pred.get(go_term, 0.0)
            deep_score = deep_pred.get(go_term, 0.0)
            
            combined_score = alpha * sim_score + (1 - alpha) * deep_score
            combined_preds[protein_id][go_term] = combined_score
    
    return dict(combined_preds)

def predict_protein_functions_deepgozero(fasta_file, output_dir, threshold=0.3,
                                       model_path=None, terms_file=None,
                                       train_data_file=None, use_diamond=True):
    """
    Predict protein functions using DeepGOZero
    
    Args:
        fasta_file (str): Path to input FASTA file
        output_dir (str): Output directory
        threshold (float): Prediction threshold
        model_path (str): Path to DeepGOZero model
        terms_file (str): Path to GO terms file
        train_data_file (str): Path to training data file
        use_diamond (bool): Whether to use Diamond similarity search
    
    Returns:
        pd.DataFrame: Prediction results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Gene Ontology
    logging.info("Loading Gene Ontology...")
    ontology = Ontology(GO_OBO_FILE, with_rels=True)
    
    predictions = {}
    
    # Diamond similarity predictions
    if use_diamond:
        try:
            diamond_output = os.path.join(output_dir, "diamond_results.tsv")
            run_diamond_search(fasta_file, DIAMOND_DB, diamond_output)
            
            # Load training annotations
            if train_data_file and os.path.exists(train_data_file):
                annotations = load_training_annotations(train_data_file)
                similarity_preds = predict_with_similarity(diamond_output, annotations, ontology)
                predictions['similarity'] = similarity_preds
        except Exception as e:
            logging.error(f"Diamond prediction failed: {e}")
            predictions['similarity'] = {}
    
    # DeepGOZero predictions
    if model_path and terms_file:
        try:
            deepgozero_preds = predict_with_deepgozero(fasta_file, model_path, terms_file, ontology)
            predictions['deepgozero'] = deepgozero_preds
        except Exception as e:
            logging.error(f"DeepGOZero prediction failed: {e}")
            predictions['deepgozero'] = {}
    
    # Combine predictions if both are available
    if 'similarity' in predictions and 'deepgozero' in predictions:
        combined_preds = combine_predictions(predictions['similarity'], predictions['deepgozero'])
    elif 'similarity' in predictions:
        combined_preds = predictions['similarity']
    elif 'deepgozero' in predictions:
        combined_preds = predictions['deepgozero']
    else:
        logging.error("No predictions generated")
        return pd.DataFrame()
    
    # Process results
    results_data = []
    
    for protein_id, go_predictions in combined_preds.items():
        # Filter predictions above threshold
        filtered_preds = {go_term: score for go_term, score in go_predictions.items() 
                         if score >= threshold}
        
        if filtered_preds:
            # Sort by score
            sorted_preds = sorted(filtered_preds.items(), key=lambda x: x[1], reverse=True)
            
            go_terms = [item[0] for item in sorted_preds]
            scores = [item[1] for item in sorted_preds]
            
            # Get GO term names and namespaces
            go_names = []
            go_namespaces = []
            
            for go_term in go_terms:
                if ontology.has_term(go_term):
                    go_names.append(ontology.get_term_name(go_term))
                    go_namespaces.append(ontology.get_namespace(go_term))
                else:
                    go_names.append("Unknown")
                    go_namespaces.append("Unknown")
            
            results_data.append({
                'Protein_ID': protein_id,
                'GO_Terms': ';'.join(go_terms),
                'GO_Names': ';'.join(go_names),
                'GO_Namespaces': ';'.join(go_namespaces),
                'Prediction_Scores': ';'.join([f"{score:.4f}" for score in scores]),
                'Num_Predictions': len(go_terms)
            })
        else:
            results_data.append({
                'Protein_ID': protein_id,
                'GO_Terms': '',
                'GO_Names': '',
                'GO_Namespaces': '',
                'Prediction_Scores': '',
                'Num_Predictions': 0
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save results
    output_file = os.path.join(output_dir, f'deepgozero_predictions_th{threshold}.csv')
    results_df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
    
    # Save summary statistics
    summary_stats = {
        'Total_Proteins': len(results_df),
        'Threshold_Used': threshold,
        'Proteins_with_Predictions': sum(1 for x in results_df['Num_Predictions'] if x > 0),
        'Proteins_without_Predictions': sum(1 for x in results_df['Num_Predictions'] if x == 0),
        'Average_Predictions_per_Protein': results_df['Num_Predictions'].mean(),
        'Max_Predictions_per_Protein': results_df['Num_Predictions'].max(),
        'Min_Predictions_per_Protein': results_df['Num_Predictions'].min()
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = os.path.join(output_dir, f'deepgozero_summary_th{threshold}.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print(f"\n=== DeepGOZero Prediction Summary ===")
    print(f"Total proteins processed: {summary_stats['Total_Proteins']}")
    print(f"Threshold used: {threshold}")
    print(f"Proteins with predictions: {summary_stats['Proteins_with_Predictions']}")
    print(f"Proteins without predictions: {summary_stats['Proteins_without_Predictions']}")
    print(f"Average predictions per protein: {summary_stats['Average_Predictions_per_Protein']:.2f}")
    print(f"Results saved to: {output_dir}")
    
    return results_df

def main():
    """
    Main function to run DeepGOZero predictions
    """
    parser = argparse.ArgumentParser(description='DeepGOZero Protein Function Prediction')
    parser.add_argument('--fasta', default=DEFAULT_FASTA, help='Input FASTA file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.3, help='Prediction threshold')
    parser.add_argument('--model', help='Path to DeepGOZero model file')
    parser.add_argument('--terms', help='Path to GO terms file')
    parser.add_argument('--train-data', help='Path to training data file')
    parser.add_argument('--no-diamond', action='store_true', help='Skip Diamond similarity search')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.fasta):
        logging.error(f"Input FASTA file not found: {args.fasta}")
        return
    
    # Set default paths if not provided
    if not args.model:
        args.model = os.path.join(DEEPGOZERO_PATH, "models/deepgozero_model.pth")
    if not args.terms:
        args.terms = os.path.join(DEEPGOZERO_PATH, "data/terms.pkl")
    if not args.train_data:
        args.train_data = os.path.join(DEEPGOZERO_PATH, "data/train_data.pkl")
    
    # Run predictions
    logging.info("Starting DeepGOZero protein function prediction...")
    predictions = predict_protein_functions_deepgozero(
        fasta_file=args.fasta,
        output_dir=args.output,
        threshold=args.threshold,
        model_path=args.model,
        terms_file=args.terms,
        train_data_file=args.train_data,
        use_diamond=not args.no_diamond
    )
    
    logging.info("Prediction completed successfully!")
    logging.info(f"Check output directory: {args.output}")

if __name__ == '__main__':
    main()