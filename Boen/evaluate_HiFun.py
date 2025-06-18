#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified HiFun Protein Function Prediction Script
Just plug in input FASTA, get output predictions
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse

# HiFun repository path
HIFUN_PATH = "/data/shared/tools/HiFun"
sys.path.append(HIFUN_PATH)

# Import HiFun components
from keras.models import load_model
from utility import load_fasta, blosum_embedding, word2vec_embedding
from models import focal_loss, auc_tensor
from keras_self_attention import SeqSelfAttention

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def run_hifun_prediction(fasta_file, output_dir, threshold=0.20):
    """
    Simple HiFun prediction - input FASTA, output CSV
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading sequences from {fasta_file}...")
    protein_id, protein_seq, protein_len = load_fasta(in_fasta=fasta_file)
    print(f"Loaded {len(protein_id)} sequences")
    
    print("Loading HiFun model and database...")
    # Load model components
    label_index = pd.read_pickle(os.path.join(HIFUN_PATH, 'db/goterms_level34.pkl'))
    word_index = np.load(os.path.join(HIFUN_PATH, 'db/word_index.npy'), allow_pickle=True).item()
    
    # Load model
    model = load_model(os.path.join(HIFUN_PATH, 'models/hifun_mode.h5'),
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'auc_tensor': auc_tensor,
                                       'multi_category_focal_loss2_fixed': focal_loss(gamma=2., alpha=.25)})
    
    print("Generating embeddings...")
    # Generate embeddings
    blosum_mat = blosum_embedding(protein_seq)
    word2vec_mat = word2vec_embedding(protein_seq, word_index, trim_len=1000)
    
    print("Making predictions...")
    # Predict
    predictions = model.predict([word2vec_mat, blosum_mat], verbose=0)
    
    print("Processing results...")
    # Process results
    results = []
    for i, probs in enumerate(predictions):
        # Get predictions above threshold
        above_threshold = probs >= threshold
        if np.any(above_threshold):
            indices = np.where(above_threshold)[0]
            go_terms = label_index.iloc[indices]['terms'].tolist()
            go_names = label_index.iloc[indices]['names'].tolist()
            scores = probs[indices].tolist()
            
            results.append({
                'Protein_ID': protein_id[i],
                'Sequence_Length': protein_len[i],
                'GO_Terms': ';'.join(go_terms),
                'GO_Names': ';'.join(go_names),
                'Prediction_Scores': ';'.join([f"{s:.4f}" for s in scores]),
                'Num_Predictions': len(go_terms)
            })
        else:
            results.append({
                'Protein_ID': protein_id[i],
                'Sequence_Length': protein_len[i],
                'GO_Terms': '',
                'GO_Names': '',
                'Prediction_Scores': '',
                'Num_Predictions': 0
            })
    
    # Save results
    output_file = os.path.join(output_dir, f'hifun_predictions_th{threshold}.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    total_proteins = len(results)
    with_predictions = sum(1 for r in results if r['Num_Predictions'] > 0)
    
    print(f"\n=== Results Summary ===")
    print(f"Total proteins: {total_proteins}")
    print(f"Proteins with predictions: {with_predictions}")
    print(f"Proteins without predictions: {total_proteins - with_predictions}")
    print(f"Results saved to: {output_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Simple HiFun Protein Function Prediction')
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--threshold', '-t', type=float, default=0.20, help='Prediction threshold (default: 0.20)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(HIFUN_PATH):
        print(f"Error: HiFun path not found: {HIFUN_PATH}")
        sys.exit(1)
    
    # Run prediction
    try:
        run_hifun_prediction(args.input, args.output, args.threshold)
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()