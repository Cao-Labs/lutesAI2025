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
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)

    with torch.no_grad():
        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )

    emb = output.per_residue_embedding

    if len(emb.shape) == 3:
        emb = emb[0]

    emb = emb[1:-1]

    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    return emb.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Generate ESM-3 Cosine Similarity Image")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--out", type=str, default="protein_image.png")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ESM3 model onto {DEVICE}...")

    try:
        # 🔥 IMPORTANT: free memory before loading
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Load model
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        # 🔥 CRITICAL PATCH: reduce memory usage
        if DEVICE.type == "cuda":
            client = client.half()

        print("Extracting representations...")
        embeddings = get_esm3_embeddings(args.sequence, client)
        print(f"Embedding shape: {embeddings.shape}")

        print("Calculating Cosine Similarity...")
        sim_matrix = cosine_similarity(embeddings)

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

        plt.tight_layout()
        plt.savefig(args.out, dpi=300)
        plt.close()

        print(f"Success! Saved to {args.out}")

    except torch.cuda.OutOfMemoryError:
        print("⚠️ GPU OOM — retrying on CPU...")

        DEVICE = torch.device("cpu")
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        embeddings = get_esm3_embeddings(args.sequence, client)
        sim_matrix = cosine_similarity(embeddings)

        plt.figure(figsize=(8, 8))
        sns.heatmap(sim_matrix, cmap="viridis", cbar=True)
        plt.savefig(args.out, dpi=300)
        plt.close()

        print(f"CPU fallback success! Saved to {args.out}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
