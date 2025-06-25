import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

def load_esm3_model():
    """
    Load the pre-trained ESM-3 model from Meta's official repository.
    Automatically uses GPU if available.
    """
    print("Loading ESM-3 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
    model.eval().to(torch.float32)
    return model

def generate_embedding(model, sequence):
    """
    Generate per-residue embeddings from a protein sequence using ESM-3.
    Returns a 2D tensor of shape [L, D] where:
    - L = sequence length
    - D = embedding dimension
    """
    protein = ESMProtein(sequence=sequence)
    sequence_tensor = model.encode(protein)

    with torch.no_grad():
        result = model.forward_and_sample(
            sequence_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )
        embedding = result.per_residue_embedding  # Shape: [L, D]
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding

def to_2d_matrix(embedding):
    """
    Transform a [L x D] embedding into a 2D image-like matrix.
    Pads to nearest square then averages across the sequence axis.
    Output shape: [sqrt(D), sqrt(D)]
    """
    seq_len, embed_dim = embedding.size()
    size = int(np.ceil(np.sqrt(embed_dim)))

    # Pad each row to make it square
    padded = torch.zeros((seq_len, size * size))
    padded[:, :embed_dim] = embedding

    # Reshape and average over sequence dimension
    matrix = padded.view(seq_len, size, size).mean(dim=0).numpy()
    return matrix

def normalize_matrix(matrix):
    """
    Normalize matrix values to range [0, 1] using percentile clipping
    to avoid outlier distortion in visualization.
    """
    lower = np.percentile(matrix, 1)
    upper = np.percentile(matrix, 99)
    clipped = np.clip(matrix, lower, upper)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)

def save_image(matrix, output_path):
    """
    Save a normalized protein matrix as a .png heatmap image.
    """
    normalized = normalize_matrix(matrix)
    plt.imshow(normalized, cmap='viridis')
    plt.title("Protein Feature Image (ESM-3)")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[✓] Saved image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate normalized protein feature image from raw sequence.")
    parser.add_argument("--sequence", type=str, required=True, help="Protein sequence in quotes")
    parser.add_argument("--out", type=str, default="protein_image.png", help="Output image filename")
    args = parser.parse_args()

    model = load_esm3_model()
    embedding = generate_embedding(model, args.sequence)
    matrix = to_2d_matrix(embedding)
    save_image(matrix, args.out)

if __name__ == "__main__":
    main()
