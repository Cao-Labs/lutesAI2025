# generate_protein_image.py

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ESM-3 imports (protein language model)
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


def normalize(mat):
    """
    FIXED:
    - Removed gamma correction (np.power)
    - Keeps ONLY min-max scaling to avoid color exaggeration
    """
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)
    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--out", type=str, default="protein_image.png")
    args = parser.parse_args()

    DEVICE = torch.device('cpu')
    print(f"Loading ESM-3 on {DEVICE}")

    try:
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        if DEVICE.type == "cuda":
            client = client.half()

        embeddings = get_esm3_embeddings(args.sequence, client)
        print("Embedding shape:", embeddings.shape)

        # ----------------------------------------
        # STEP 2: Representations
        # ----------------------------------------

        sim_matrix = cosine_similarity(embeddings)

        dist_matrix = np.linalg.norm(
            embeddings[:, None] - embeddings[None, :],
            axis=-1
        )

        sim_matrix = normalize(sim_matrix)
        dist_matrix = normalize(dist_matrix)

        # ----------------------------------------
        # STEP 3: Combine signals (KEEP THIS)
        # ----------------------------------------

        combined = (sim_matrix + (1 - dist_matrix)) / 2

        # ----------------------------------------
        # STEP 4: REMOVED EDGE ENHANCEMENT
        # ----------------------------------------
        # (This was causing BLIP-2 to focus on artificial edges/colors)

        # ----------------------------------------
        # STEP 5: Visualization (FIXED COLORMAP)
        # ----------------------------------------

        plt.figure(figsize=(6, 6))

        # FIX: inferno → viridis (reduces color distraction)
        sns.heatmap(combined, cmap="viridis", cbar=True)

        plt.title("Protein Structural Similarity")
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.savefig(args.out, dpi=300)
        plt.close()

        print(f"Saved → {args.out}")

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
