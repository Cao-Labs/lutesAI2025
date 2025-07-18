# -*- coding: utf-8 -*-
"""
Custom HiFun prediction script for protein function prediction
Modified to run on specific FASTA file and output to specified directory.

*** CORRECTION: This version saves ALL predictions to the evaluation file,
*** without any pre-filtering, to ensure the evaluation script has
*** complete data and can calculate recall correctly.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import shutil
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
    
    # Clean the output directory to ensure a fresh start
    logger.info(f"Preparing a clean output directory at: {output_dir}")
    if os.path.exists(output_dir):
        logger.warning("Output directory exists. Removing it to ensure a clean run.")
        shutil.rmtree(output_dir)
    
    # Create the fresh output directory
    os.makedirs(output_dir)
    logger.info("Output directory created.")
    
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
        
        required_files = [
            'db/goterms_level34.pkl', 'db/word_index.npy', 'models/hifun_mode.h5'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load model components
        label_index = pd.read_pickle('db/goterms_level34.pkl')
        word_index = np.load('db/word_index.npy', allow_pickle=True).item()
        
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
        
        # --- Save results in the "long" format for the evaluation script ---
        eval_output_file = os.path.join(output_dir, 'predictions_for_eval.txt')
        logger.info(f"Saving ALL predictions in evaluation format to: {eval_output_file}")
        
        with open(eval_output_file, 'w') as f:
            f.write("AUTHOR G_Gemini\nMODEL 1\nKEYWORDS deep learning\n")
            # Iterate through each protein and its prediction scores
            for i, prot_id in enumerate(protein_id):
                probs = predict_probs[i]
                for j, go_term in enumerate(label_index['terms']):
                    score = probs[j]
                    # --- FIX ---
                    # Write ALL predictions, no matter how low the score.
                    # The evaluation script will handle the thresholding.
                    f.write(f"{prot_id}\t{go_term}\t{score:.5f}\n")
            f.write("END\n")

        logger.info("Evaluation format file saved.")

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
        predict_protein_functions()
        print("Prediction completed successfully!")
        print(f"Results saved to: /data/summer2020/Boen/hifun_predictions/")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
