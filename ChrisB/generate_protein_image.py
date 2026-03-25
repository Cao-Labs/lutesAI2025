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
    """
    Extract per-residue embeddings from ESM-3.

    These embeddings encode structural and biochemical information
    learned from large-scale protein sequence data.

    Output shape: (L, D)
    where L = sequence length, D = embedding dimension
    """

    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)

    with torch.no_grad():
        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )

    emb = output.per_residue_embedding

    # Handle batch dimension if present
    if len(emb.shape) == 3:
        emb = emb[0]

    # Remove BOS/EOS tokens (not biologically meaningful residues)
    emb = emb[1:-1]

    # Replace NaNs/Infs to ensure stable downstream computation
    emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    return emb.cpu().numpy()


def normalize(mat):
    """
    Normalize matrix for visualization.

    IMPORTANT:
    - Min-max normalization preserves relative structure
      (better than z-score for visual models)
    - Gamma correction enhances contrast for pattern detection

    This improves interpretability for vision-language models (BLIP-2),
    which rely on visual contrast rather than numeric scale.
    """

    # Scale to [0, 1]
    mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)

    # Boost contrast (emphasizes domains/diagonals)
    mat = np.power(mat, 0.5)

    return mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--out", type=str, default="protein_image.png")
    args = parser.parse_args()

    # Use GPU if available (optional, CPU is fine)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ESM-3 on {DEVICE}")

    try:
        # Load pretrained ESM-3 model
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=DEVICE)
        client.eval()

        # Half precision for GPU efficiency
        if DEVICE.type == "cuda":
            client = client.half()

        # Step 1: Get residue-level embeddings
        embeddings = get_esm3_embeddings(args.sequence, client)
        print("Embedding shape:", embeddings.shape)

        # --------------------------------------------------
        # STEP 2: Construct structural representations
        # --------------------------------------------------

        # Cosine similarity → captures relational similarity between residues
        sim_matrix = cosine_similarity(embeddings)

        # Euclidean distance → captures absolute differences
        dist_matrix = np.linalg.norm(
            embeddings[:, None] - embeddings[None, :],
            axis=-1
        )

        # Normalize both for visual consistency
        sim_matrix = normalize(sim_matrix)
        dist_matrix = normalize(dist_matrix)

        # --------------------------------------------------
        # STEP 3: Combine signals into ONE image
        # --------------------------------------------------

        """
        IMPORTANT DESIGN DECISION:

        We combine similarity and distance into a single matrix.

        Why?
        - Vision-language models (e.g., BLIP-2) are NOT trained on
          multi-panel scientific figures
        - Single-image representation reduces ambiguity
        - Preserves both relational and absolute structure signals

        This avoids confusion seen in multi-heatmap layouts.
        """

        combined = (sim_matrix + (1 - dist_matrix)) / 2

        # --------------------------------------------------
        # STEP 4: Enhance structural boundaries
        # --------------------------------------------------

        """
        Edge enhancement highlights domain boundaries and transitions.

        This improves:
        - detection of structural motifs
        - interpretability by vision models

        Safe optional step (fallback if scipy not installed)
        """

        try:
            import scipy.ndimage as ndi
            edges = ndi.sobel(sim_matrix)

            combined = combined + 0.2 * edges
            combined = np.clip(combined, 0, 1)

        except ImportError:
            pass  # safe fallback

        # --------------------------------------------------
        # STEP 5: Visualization
        # --------------------------------------------------

        """
        Visualization choices are CRITICAL:

        - Use a single heatmap (not multiple panels)
        - Use high-contrast colormap ("inferno")
        - Remove axis ticks (reduce noise)
        - Keep resolution high (dpi=300)

        These choices are optimized for BLIP-2 interpretation,
        not human scientific plotting.
        """

        plt.figure(figsize=(6, 6))
        sns.heatmap(combined, cmap="inferno", cbar=True)

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
