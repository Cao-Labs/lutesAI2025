# -*- coding: utf-8 -*-
"""
HiFun Protein Function Prediction Benchmark Script
Modified for benchmarking against testing sequences
"""
import os
import logging
import pandas as pd
import numpy as np
from keras.models import load_model
from utility import load_fasta, blosum_embedding, word2vec_embedding
from models import focal_loss, auc_tensor
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import argparse

# Configure environment and logging
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
logging.basicConfig(level=logging.INFO)

# Input and Output paths
FASTA_FILE = "/data/summer2020/naufal/testing_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/Boen/output"

def predict_protein_functions(in_fasta, output_dir, threshold=0.20):
    """
    Predict protein functions using HiFun model
    
    Args:
        in_fasta (str): Path to input FASTA file
        output_dir (str): Directory to save results
        threshold (float): Prediction threshold
    
    Returns:
        pd.DataFrame: Prediction results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Loading sequences from {in_fasta}")
    # Load query proteins
    protein_id, protein_seq, protein_len = load_fasta(in_fasta=in_fasta)
    logging.info(f"Loaded {len(protein_id)} protein sequences")
    
    # Load pre-built models and data
    logging.info("Loading model components...")
    label_index = pd.read_pickle('db/goterms_level34.pkl')
    word_index = np.load('db/word_index.npy', allow_pickle=True).item()
    
    # Load the trained model
    model = load_model('models/hifun_mode.h5',
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'auc_tensor': auc_tensor,
                                       'multi_category_focal_loss2_fixed': focal_loss(gamma=2., alpha=.25)})
    
    embeddings_matrix = np.load("db/embeddings_matrix.npy")
    
    # Generate embedding matrices
    logging.info("Generating protein embeddings...")
    blosum_mat = blosum_embedding(protein_seq)
    word2vec_mat = word2vec_embedding(protein_seq, word_index, trim_len=1000)
    
    # Make predictions
    logging.info("Making predictions...")
    predict_probs = model.predict([word2vec_mat, blosum_mat], verbose=1)
    
    # Process predictions
    predict_terms = []
    predict_names = []
    predict_levels = []
    predict_binary = []
    
    for prob in predict_probs:
        ind = np.argwhere(prob >= threshold).flatten().tolist()
        predict_terms.append(';'.join(label_index.iloc[ind, 0].to_list()))
        predict_names.append(';'.join(label_index.iloc[ind, 1].to_list()))
        predict_levels.append(';'.join(map(str, label_index.iloc[ind, 2].to_list())))
        
        # Create binary prediction vector for evaluation
        binary_pred = np.zeros(len(label_index))
        binary_pred[ind] = 1
        predict_binary.append(binary_pred)
    
    # Create results dataframe
    predict_res = pd.DataFrame({
        'Protein_id': protein_id,
        'GO_terms': predict_terms,
        'GO_names': predict_names,
        'GO_levels': predict_levels
    })
    
    # Add probability scores for each GO term
    prob_df = pd.DataFrame(predict_probs, columns=label_index['terms'])
    predict_res = pd.concat([predict_res, prob_df], axis=1)
    
    # Save results
    output_file = os.path.join(output_dir, f'hifun_predictions_th{threshold}.csv')
    predict_res.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
    
    # Save binary predictions for evaluation
    binary_predictions = np.array(predict_binary)
    np.save(os.path.join(output_dir, f'binary_predictions_th{threshold}.npy'), binary_predictions)
    
    return predict_res, binary_predictions, label_index

def evaluate_predictions(true_labels, predicted_labels, go_terms, output_dir, threshold):
    """
    Evaluate predictions using precision, recall, and F1-score
    
    Args:
        true_labels (np.array): True binary labels
        predicted_labels (np.array): Predicted binary labels
        go_terms (list): List of GO terms
        output_dir (str): Directory to save evaluation results
        threshold (float): Prediction threshold used
    """
    logging.info("Evaluating predictions...")
    
    # Calculate metrics
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    precision_micro = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
    recall_micro = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
    f1_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
    
    # Create evaluation summary
    eval_summary = {
        'Threshold': threshold,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1-Score (Macro)': f1_macro,
        'Precision (Micro)': precision_micro,
        'Recall (Micro)': recall_micro,
        'F1-Score (Micro)': f1_micro,
        'Total Samples': len(true_labels),
        'Total GO Terms': len(go_terms)
    }
    
    # Save evaluation results
    eval_df = pd.DataFrame([eval_summary])
    eval_file = os.path.join(output_dir, f'evaluation_results_th{threshold}.csv')
    eval_df.to_csv(eval_file, index=False)
    
    # Generate detailed classification report
    report = classification_report(true_labels, predicted_labels, 
                                 target_names=go_terms, 
                                 output_dict=True, 
                                 zero_division=0)
    
    report_df = pd.DataFrame(report).transpose()
    report_file = os.path.join(output_dir, f'detailed_classification_report_th{threshold}.csv')
    report_df.to_csv(report_file)
    
    logging.info(f"Evaluation results saved to {eval_file}")
    logging.info(f"Detailed report saved to {report_file}")
    
    # Print summary
    print(f"\n=== HiFun Benchmark Results (Threshold: {threshold}) ===")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Micro): {precision_micro:.4f}")
    print(f"Recall (Micro): {recall_micro:.4f}")
    print(f"F1-Score (Micro): {f1_micro:.4f}")
    
    return eval_summary

def load_true_labels(label_file, protein_ids, go_terms):
    """
    Load true labels for evaluation
    
    Args:
        label_file (str): Path to file containing true labels
        protein_ids (list): List of protein IDs
        go_terms (list): List of GO terms
    
    Returns:
        np.array: Binary matrix of true labels
    """
    # This function needs to be implemented based on your true label format
    # Example implementation:
    logging.info(f"Loading true labels from {label_file}")
    
    # Placeholder - you'll need to modify this based on your label format
    # For now, return random labels as example
    true_labels = np.random.randint(0, 2, size=(len(protein_ids), len(go_terms)))
    
    logging.warning("Using random labels as placeholder. Please implement load_true_labels function.")
    return true_labels

def main():
    """
    Main function to run HiFun benchmark
    """
    parser = argparse.ArgumentParser(description='HiFun Protein Function Prediction Benchmark')
    parser.add_argument('--fasta', default=FASTA_FILE, help='Input FASTA file')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.20, help='Prediction threshold')
    parser.add_argument('--true_labels', default=None, help='File containing true labels for evaluation')
    
    args = parser.parse_args()
    
    # Run predictions
    predictions, binary_predictions, label_index = predict_protein_functions(
        args.fasta, args.output, args.threshold
    )
    
    # If true labels are provided, evaluate predictions
    if args.true_labels:
        protein_ids = predictions['Protein_id'].tolist()
        go_terms = label_index['terms'].tolist()
        
        true_labels = load_true_labels(args.true_labels, protein_ids, go_terms)
        
        evaluation = evaluate_predictions(
            true_labels, binary_predictions, go_terms, args.output, args.threshold
        )
    else:
        logging.info("No true labels provided. Skipping evaluation.")
        logging.info("To evaluate predictions, provide --true_labels argument")
    
    logging.info("Benchmark completed successfully!")

if __name__ == '__main__':
    main()