# -*- coding: utf-8 -*-
"""
Custom HiFun prediction script for protein function prediction
Modified to run on specific FASTA file and output to specified directory
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from keras.models import load_model
from keras_self_attention import SeqSelfAttention

# Add HiFun directory to Python path
sys.path.append('/data/shared/tools/HiFun')

# Import HiFun modules
from utility import load_fasta, blosum_embedding, word2vec_embedding
from models import focal_loss, auc_tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def predict_protein_functions():
    """
    Run HiFun predictions on testing_sequences.fasta
    """
    # File paths
    input_fasta = '/data/summer2020/Boen/benchmark_testing_sequences.fasta'
    output_dir = '/data/summer2020/Boen/hifun_predictions'
    hifun_dir = '/data/shared/tools/HiFun'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to HiFun directory to access relative paths
    original_dir = os.getcwd()
    os.chdir(hifun_dir)
    
    try:
        logger.info(f"Loading proteins from: {input_fasta}")
        
        # Check if input file exists
        if not os.path.exists(input_fasta):
            raise FileNotFoundError(f"Input FASTA file not found: {input_fasta}")
        
        # Load query proteins
        protein_id, protein_seq, protein_len = load_fasta(input_fasta)
        logger.info(f"Loaded {len(protein_id)} proteins")
        
        # Load pre-built models and data
        logger.info("Loading model components...")
        
        # Check for required files
        required_files = [
            'db/goterms_level34.pkl',
            'db/word_index.npy',
            'models/hifun_mode.h5',
            'db/embeddings_matrix.npy'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load model components
        label_index = pd.read_pickle('db/goterms_level34.pkl')
        word_index = np.load('db/word_index.npy', allow_pickle=True).item()
        embeddings_matrix = np.load("db/embeddings_matrix.npy")
        
        logger.info("Loading trained model...")
        model = load_model('models/hifun_mode.h5',
                          custom_objects={
                              'SeqSelfAttention': SeqSelfAttention,
                              'auc_tensor': auc_tensor,
                              'multi_category_focal_loss2_fixed': focal_loss(gamma=2., alpha=.25)
                          })
        
        # Generate embedding matrices
        logger.info("Generating protein embeddings...")
        blosum_mat = blosum_embedding(protein_seq)
        word2vec_mat = word2vec_embedding(protein_seq, word_index, trim_len=1000)
        
        logger.info("Running predictions...")
        # Make predictions
        predict_probs = model.predict([word2vec_mat, blosum_mat], verbose=1)
        
        # Process predictions with threshold
        threshold = 0.20
        predict_terms = []
        predict_names = []
        predict_levels = []
        
        logger.info(f"Processing predictions with threshold {threshold}...")
        for prob in predict_probs:
            ind = np.argwhere(prob >= threshold).flatten().tolist()
            if len(ind) > 0:
                predict_terms.append(';'.join(label_index.iloc[ind, 0].to_list()))
                predict_names.append(';'.join(label_index.iloc[ind, 1].to_list()))
                predict_levels.append(';'.join(map(str, label_index.iloc[ind, 2].to_list())))
            else:
                predict_terms.append('')
                predict_names.append('')
                predict_levels.append('')
        
        # Create results dataframe
        predict_res = pd.DataFrame({
            'Protein_id': protein_id,
            'GO_terms': predict_terms,
            'GO_names': predict_names,
            'GO_levels': predict_levels
        })
        
        # Add probability scores for all GO terms
        prob_df = pd.DataFrame(predict_probs, columns=label_index['terms'])
        predict_res = pd.concat([predict_res, prob_df], axis=1)
        
        # Save results
        output_file = os.path.join(output_dir, 'hifun_predictions.csv')
        predict_res.to_csv(output_file, index=False)
        
        logger.info(f"Predictions saved to: {output_file}")
        logger.info(f"Processed {len(protein_id)} proteins")
        logger.info(f"Results shape: {predict_res.shape}")
        
        # Save a summary file as well
        summary_file = os.path.join(output_dir, 'prediction_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"HiFun Protein Function Prediction Results\n")
            f.write(f"=========================================\n\n")
            f.write(f"Input file: {input_fasta}\n")
            f.write(f"Number of proteins processed: {len(protein_id)}\n")
            f.write(f"Prediction threshold: {threshold}\n")
            f.write(f"Output file: {output_file}\n")
            f.write(f"Results dimensions: {predict_res.shape[0]} rows x {predict_res.shape[1]} columns\n\n")
            
            # Count proteins with predictions
            proteins_with_predictions = sum(1 for terms in predict_terms if terms != '')
            f.write(f"Proteins with predictions above threshold: {proteins_with_predictions}\n")
            f.write(f"Proteins without predictions: {len(protein_id) - proteins_with_predictions}\n")
        
        logger.info(f"Summary saved to: {summary_file}")
        
        return predict_res
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
    finally:
        # Return to original directory
        os.chdir(original_dir)

def main():
    """
    Main function to run the prediction pipeline
    """
    try:
        results = predict_protein_functions()
        print("Prediction completed successfully!")
        print(f"Results saved to: /data/summer2020/Boen/output/")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()