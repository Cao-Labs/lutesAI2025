# -*- coding: utf-8 -*-
"""
HiFun Protein Function Prediction Script
Simplified version for GO term predictions only
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

# HiFun repository path
HIFUN_PATH = "/data/shared/tools/HiFun"
sys.path.append(HIFUN_PATH)

from keras.models import load_model
from utility import load_fasta, blosum_embedding, word2vec_embedding
from models import focal_loss, auc_tensor
from keras_self_attention import SeqSelfAttention
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
    label_index = pd.read_pickle(os.path.join(HIFUN_PATH, 'db/goterms_level34.pkl'))
    word_index = np.load(os.path.join(HIFUN_PATH, 'db/word_index.npy'), allow_pickle=True).item()
    
    # Load the trained model
    logging.info("Loading HiFun model...")
    model = load_model(os.path.join(HIFUN_PATH, 'models/hifun_mode.h5'),
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'auc_tensor': auc_tensor,
                                       'multi_category_focal_loss2_fixed': focal_loss(gamma=2., alpha=.25)})
    
    embeddings_matrix = np.load(os.path.join(HIFUN_PATH, "db/embeddings_matrix.npy"))
    
    # Generate embedding matrices
    logging.info("Generating protein embeddings...")
    blosum_mat = blosum_embedding(protein_seq)
    word2vec_mat = word2vec_embedding(protein_seq, word_index, trim_len=1000)
    
    # Make predictions
    logging.info("Making GO term predictions...")
    predict_probs = model.predict([word2vec_mat, blosum_mat], verbose=1)
    
    # Process predictions
    predict_terms = []
    predict_names = []
    predict_levels = []
    predict_scores = []
    
    for i, prob in enumerate(predict_probs):
        # Get indices where probability >= threshold
        ind = np.argwhere(prob >= threshold).flatten().tolist()
        
        if len(ind) > 0:
            # Get GO terms, names, and levels for predictions above threshold
            terms = label_index.iloc[ind, 0].to_list()
            names = label_index.iloc[ind, 1].to_list()
            levels = label_index.iloc[ind, 2].to_list()
            scores = prob[ind].tolist()
            
            predict_terms.append(';'.join(terms))
            predict_names.append(';'.join(names))
            predict_levels.append(';'.join(map(str, levels)))
            predict_scores.append(';'.join([f"{score:.4f}" for score in scores]))
        else:
            # No predictions above threshold
            predict_terms.append('')
            predict_names.append('')
            predict_levels.append('')
            predict_scores.append('')
    
    # Create results dataframe
    predict_res = pd.DataFrame({
        'Protein_ID': protein_id,
        'Sequence_Length': protein_len,
        'GO_Terms': predict_terms,
        'GO_Names': predict_names,
        'GO_Levels': predict_levels,
        'Prediction_Scores': predict_scores,
        'Num_Predictions': [len(terms.split(';')) if terms else 0 for terms in predict_terms]
    })
    
    # Save main results
    output_file = os.path.join(output_dir, f'hifun_predictions_th{threshold}.csv')
    predict_res.to_csv(output_file, index=False)
    logging.info(f"Main predictions saved to {output_file}")
    
    # Save detailed probability matrix (optional - includes all GO term probabilities)
    prob_df = pd.DataFrame(predict_probs, columns=label_index['terms'])
    prob_df.insert(0, 'Protein_ID', protein_id)
    detailed_output_file = os.path.join(output_dir, f'hifun_detailed_probabilities_th{threshold}.csv')
    prob_df.to_csv(detailed_output_file, index=False)
    logging.info(f"Detailed probabilities saved to {detailed_output_file}")
    
    # Save summary statistics
    summary_stats = {
        'Total_Proteins': len(protein_id),
        'Threshold_Used': threshold,
        'Proteins_with_Predictions': sum(1 for terms in predict_terms if terms),
        'Proteins_without_Predictions': sum(1 for terms in predict_terms if not terms),
        'Total_GO_Terms_Available': len(label_index),
        'Average_Predictions_per_Protein': np.mean([len(terms.split(';')) if terms else 0 for terms in predict_terms]),
        'Max_Predictions_per_Protein': max([len(terms.split(';')) if terms else 0 for terms in predict_terms]),
        'Min_Predictions_per_Protein': min([len(terms.split(';')) if terms else 0 for terms in predict_terms])
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = os.path.join(output_dir, f'hifun_summary_th{threshold}.csv')
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Summary statistics saved to {summary_file}")
    
    # Print summary to console
    print(f"\n=== HiFun Prediction Summary ===")
    print(f"Total proteins processed: {summary_stats['Total_Proteins']}")
    print(f"Threshold used: {threshold}")
    print(f"Proteins with predictions: {summary_stats['Proteins_with_Predictions']}")
    print(f"Proteins without predictions: {summary_stats['Proteins_without_Predictions']}")
    print(f"Average predictions per protein: {summary_stats['Average_Predictions_per_Protein']:.2f}")
    print(f"Results saved to: {output_dir}")
    
    return predict_res

def main():
    """
    Main function to run HiFun predictions
    """
    parser = argparse.ArgumentParser(description='HiFun Protein Function Prediction')
    parser.add_argument('--fasta', default=FASTA_FILE, help='Input FASTA file')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.20, help='Prediction threshold')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.fasta):
        logging.error(f"Input FASTA file not found: {args.fasta}")
        return
    
    # Check if HiFun path exists
    if not os.path.exists(HIFUN_PATH):
        logging.error(f"HiFun repository not found at: {HIFUN_PATH}")
        return
    
    # Run predictions
    logging.info("Starting HiFun protein function prediction...")
    predictions = predict_protein_functions(args.fasta, args.output, args.threshold)
    
    logging.info("Prediction completed successfully!")
    logging.info(f"Check output directory: {args.output}")

if __name__ == '__main__':
    main()