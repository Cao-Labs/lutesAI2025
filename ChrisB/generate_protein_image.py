import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# ---- FIX: force transformers to avoid fast tokenizer conflicts ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL


# -----------------------------
# Load ESM-3 model
# -----------------------------
def load_esm3_model():

    print("[INFO] Loading ESM-3 model...")

    model = ESM3.from_pretrained(ESM3_OPEN_SMALL)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    return model


# -----------------------------
# Generate embedding
# -----------------------------
def generate_embedding(model, sequence):

    protein = ESMProtein(sequence=sequence)

    config = SamplingConfig(
        return_per_residue_embeddings=True
    )

    with torch.no_grad():
        result = model.generate(protein, config)

    embedding = result.per_residue_embedding

    # remove NaNs
    embedding = torch.nan_to_num(
        embedding,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return embedding   # (L, D)


# -----------------------------
# Convert embedding → similarity matrix
# -----------------------------
def to_similarity_matrix(embedding):

    embedding_np = embedding.cpu().numpy()

    sim_matrix = cosine_similarity(embedding_np)

    return sim_matrix


# -----------------------------
# Normalize matrix
# -----------------------------
def normalize_matrix(matrix):

    min_val = matrix.min()
    max_val = matrix.max()

    normalized = (matrix - min_val) / (max_val - min_val + 1e-8)

    return normalized


# -----------------------------
# Save PNG
# -----------------------------
def save_image(matrix, output_path):

    normalized = normalize_matrix(matrix)

    plt.figure(figsize=(5,5))

    plt.imshow(
        normalized,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.title("Protein Similarity Image (ESM-3)")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight")

    plt.close()

    print(f"[✓] Saved image: {output_path}")


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Generate protein feature image from sequence using ESM-3."
    )

    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Protein sequence"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="protein_image.png",
        help="Output image filename"
    )

    args = parser.parse_args()

    model = load_esm3_model()

    embedding = generate_embedding(model, args.sequence)

    matrix = to_similarity_matrix(embedding)

    save_image(matrix, args.out)


if __name__ == "__main__":
    main()
