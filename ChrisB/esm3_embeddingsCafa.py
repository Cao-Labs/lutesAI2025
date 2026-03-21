# generate_protein_image.py
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ESM3 Specific Imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

def get_esm3_embeddings(seq: str, client: ESM3) -> np.ndarray:
    """
    Extracts per-residue embeddings and safely removes special tokens.
    """
    # 1. Wrap sequence into the required ESMProtein payload
    protein = ESMProtein(sequence=seq)
    
    # 2. Encode to tensor (this automatically adds BOS and EOS tokens)
    protein_tensor = client.encode(protein)
    
    # 3. Forward pass with Generative Override (gradient tracking MUST be off)
    with torch.no_grad():
        output = client.forward_and_sample(
            protein_tensor, 
            SamplingConfig(return_per_residue_embeddings=True)
        )
        
    # 4. Extract embeddings safely
    emb = output.per_residue_embedding

    # PATCH: handle possible batch dimension
    if len(emb.shape) == 3:
        emb = emb[0]

    # Remove BOS/EOS tokens
    emb = emb[1:-1]

    # PATCH: clean numerical issues
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    return emb.cpu().numpy()

def main():
    # Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Generate ESM-3 Cosine Similarity Image")
    parser.add_argument("--sequence", type=str, required=True, help="Raw amino acid sequence")
    parser.add_argument("--out", type=str, default="protein_image.png", help="Output filename")
    args = parser.parse_args()

    # Dynamic Device Mapping
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading 1.4B Parameter ESM3 Model onto {DEVICE}...")
    
    try:
        # Load Model
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()
        
        # Extract Clean Embeddings
        print("Extracting representations...")
        embeddings = get_esm3_embeddings(args.sequence, client)
        print(f"Extracted biological embedding matrix shape: {embeddings.shape}")
        
        # Calculate Cosine Similarity
        print("Calculating Cosine Similarity matrix...")
        sim_matrix = cosine_similarity(embeddings)
        
        # Plotting the corrected Heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            sim_matrix, 
            cmap="viridis", 
            cbar=True, 
            xticklabels=False, 
            yticklabels=False,
            vmin=0.0,
            vmax=1.0
        )
        plt.title("Protein Feature Image (ESM-3 Cosine Similarity) - Artifacts Removed")
        
        # Save output
        plt.tight_layout()
        plt.savefig(args.out, dpi=300)
        plt.close()
        
        print(f"Success! Clean image saved to {args.out}")

    except Exception as e:
        print(f"Hardware or execution error occurred: {e}")

if __name__ == "__main__":
    main()
