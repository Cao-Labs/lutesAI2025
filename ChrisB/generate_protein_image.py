# generate_protein_image.py

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ESM-3 model imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL


def get_esm3_embeddings(seq: str, client: ESM3) -> np.ndarray:
    """
    Convert a protein sequence into per-residue embeddings using ESM-3.

    Steps:
    1. Wrap sequence into ESMProtein object
    2. Encode sequence (adds special tokens)
    3. Run model to get embeddings
    4. Clean output (remove extra tokens + fix values)
    """

    # Step 1: Wrap raw sequence
    protein = ESMProtein(sequence=seq)

    # Step 2: Convert to model input format (adds BOS/EOS tokens)
    protein_tensor = client.encode(protein)

    # Step 3: Run model (no gradients needed)
    with torch.no_grad():
        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )

    # Extract embeddings
    emb = output.per_residue_embedding

    # Handle possible batch dimension [1, L, D] -> [L, D]
    if len(emb.shape) == 3:
        emb = emb[0]

    # Remove special tokens (BOS and EOS)
    emb = emb[1:-1]

    # Replace NaN or infinite values with 0
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to NumPy array
    return emb.cpu().numpy()


def main():
    """
    Main function:
    1. Load ESM-3 model
    2. Generate embeddings
    3. Compute cosine similarity
    4. Save heatmap image
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate protein image using ESM-3 and cosine similarity"
    )
    parser.add_argument("--sequence", type=str, required=True,
                        help="Protein sequence (amino acids)")
    parser.add_argument("--out", type=str, default="protein_image.png",
                        help="Output image file")
    args = parser.parse_args()

    # Select device (GPU if available, else CPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ESM-3 model on {DEVICE}...")

    try:
        # Clear GPU memory if using CUDA
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Load ESM-3 model
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        # Use half precision on GPU to reduce memory usage
        if DEVICE.type == "cuda":
            client = client.half()

        print("Extracting embeddings...")
        embeddings = get_esm3_embeddings(args.sequence, client)
        print(f"Embedding shape: {embeddings.shape}")

        # Compute similarity between residues
        print("Computing cosine similarity...")
        sim_matrix = cosine_similarity(embeddings)

        # Plot heatmap
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

        plt.title("Protein Feature Image (ESM-3 Cosine Similarity)")

        # Save image
        plt.tight_layout()
        plt.savefig(args.out, dpi=300)
        plt.close()

        print(f"Saved image to {args.out}")

    except torch.cuda.OutOfMemoryError:
        # If GPU fails, switch to CPU
        print("GPU out of memory. Switching to CPU...")

        DEVICE = torch.device("cpu")
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        embeddings = get_esm3_embeddings(args.sequence, client)
        sim_matrix = cosine_similarity(embeddings)

        plt.figure(figsize=(8, 8))
        sns.heatmap(sim_matrix, cmap="viridis", cbar=True)

        plt.savefig(args.out, dpi=300)
        plt.close()

        print(f"Saved image to {args.out} (CPU mode)")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
