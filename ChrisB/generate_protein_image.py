# generate_protein_image.py

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

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

    emb = emb[1:-1]  # remove BOS/EOS
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    return emb.cpu().numpy()


def normalize(mat):
    return (mat - mat.mean()) / (mat.std() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--out", type=str, default="protein_image.png")
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ESM-3 on {DEVICE}")

    try:
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        if DEVICE.type == "cuda":
            client = client.half()

        embeddings = get_esm3_embeddings(args.sequence, client)
        print("Embedding shape:", embeddings.shape)

        # 🔥 MULTI-SIGNAL
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = np.linalg.norm(
            embeddings[:, None] - embeddings[None, :],
            axis=-1
        )

        # 🔥 NORMALIZE
        sim_matrix = normalize(sim_matrix)
        dist_matrix = normalize(dist_matrix)

        # 🔥 PLOT SIDE-BY-SIDE
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(sim_matrix, cmap="viridis", ax=axes[0], cbar=True)
        axes[0].set_title("Cosine Similarity")

        sns.heatmap(dist_matrix, cmap="magma", ax=axes[1], cbar=True)
        axes[1].set_title("Euclidean Distance")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("Protein Representation (Multi-Channel)")
        plt.tight_layout()
        plt.savefig(args.out, dpi=300)
        plt.close()

        print(f"Saved → {args.out}")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
