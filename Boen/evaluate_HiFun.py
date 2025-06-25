# -*- coding: utf-8 -*-
"""
Custom HiFun prediction script for protein function prediction.
MODIFIED to iterate through a directory of individual FASTA files and use the
filename as the protein ID to ensure consistency with the ground truth generation script.

*** CORRECTION: The word2vec model is now pre-loaded to prevent reloading it in a loop,
*** which significantly improves performance.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import shutil
from keras.models import load_model
from keras_self_attention import SeqSelfAttention
from gensim.models.fasttext import FastText # <-- ADDED: Import for fasttext model

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
    Run HiFun predictions by iterating through a directory of FASTA files.
    """
    # File paths
    input_fasta_dir = '/data/summer2020/Boen/benchmark_testing_sequences'
    output_dir = '/data/summer2020/Boen/hifun_predictions'
    hifun_dir = '/data/shared/tools/HiFun'
    
    # Clean and create the output directory
    logger.info(f"Preparing a clean output directory at: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.info("Output directory created.")
    
    original_dir = os.getcwd()
    os.chdir(hifun_dir)
    
    try:
        # --- FIX: Load all models ONCE before the loop ---
        logger.info("Loading model components...")
        label_index = pd.read_pickle('db/goterms_level34.pkl')
        word_index = np.load('db/word_index.npy', allow_pickle=True).item()
        
        logger.info("Loading Keras prediction model...")
        model = load_model('models/hifun_mode.h5',
                          custom_objects={
                              'SeqSelfAttention': SeqSelfAttention,
                              'auc_tensor': auc_tensor,
                              'multi_category_focal_loss2_fixed': focal_loss(gamma=2., alpha=.25)
                          })

        # --- THIS IS THE CRITICAL FIX ---
        # Define path and pre-load the word2vec model to avoid reloading it in the loop
        word2vec_model_path = 'db/uniprot_sprot.bin' # Adjust this path if necessary
        logger.info(f"Pre-loading word2vec model from: {word2vec_model_path}...")
        if not os.path.exists(word2vec_model_path):
            raise FileNotFoundError(f"Word2vec model not found at: {word2vec_model_path}")
        word2vec_model = FastText.load_fasttext_format(word2vec_model_path)
        logger.info("Word2vec model loaded.")
        # ---------------------------------
        
        # Prepare the output file
        eval_output_file = os.path.join(output_dir, 'predictions_for_eval.txt')
        logger.info(f"Saving ALL predictions to: {eval_output_file}")

        with open(eval_output_file, 'w') as f:
            f.write("AUTHOR G_Gemini\nMODEL 1\nKEYWORDS deep learning\n")

            fasta_files = sorted([f for f in os.listdir(input_fasta_dir) if f.endswith(".fasta")])
            logger.info(f"Found {len(fasta_files)} FASTA files to process.")

            for i, filename in enumerate(fasta_files):
                logger.info(f"Processing file {i+1}/{len(fasta_files)}: {filename}")
                
                prot_id_from_filename = os.path.splitext(filename)[0]
                
                fasta_path = os.path.join(input_fasta_dir, filename)
                _, protein_seq, _ = load_fasta(fasta_path)
                
                if not protein_seq:
                    logger.warning(f"Could not load sequence from {filename}. Skipping.")
                    continue

                # Generate embeddings for the single protein
                blosum_mat = blosum_embedding(protein_seq)
                # Pass the pre-loaded model into the function
                word2vec_mat = word2vec_embedding(protein_seq, word_index, word2vec_model=word2vec_model, trim_len=1000)
                
                # Make predictions for this protein
                predict_probs = model.predict([word2vec_mat, blosum_mat], verbose=0)
                
                # Iterate through prediction scores and write to file
                probs_for_protein = predict_probs[0]
                for j, go_term in enumerate(label_index['terms']):
                    score = probs_for_protein[j]
                    f.write(f"{prot_id_from_filename}\t{go_term}\t{score:.5f}\n")
            
            f.write("END\n")

        logger.info("Evaluation format file saved.")

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
    finally:
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
