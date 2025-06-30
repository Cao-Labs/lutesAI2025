import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

def load_esm3_model():
    print("Loading ESM-3 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
    model.eval().to(torch.float32)
    return model

def generate_embedding(model, sequence):
    protein = ESMProtein(sequence=sequence)
    sequence_tensor = model.encode(protein)

    with torch.no_grad():
        result = model.forward_and_sample(
            sequence_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )
        embedding = result.per_residue_embedding
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding

def to_2d_matrix(embedding):
    seq_len, embed_dim = embedding.size()
    size = int(np.ceil(np.sqrt(embed_dim)))
    padded = torch.zeros((seq_len, size * size))
    padded[:, :embed_dim] = embedding
    matrix = padded.view(seq_len, size, size).mean(dim=0).numpy()
    return matrix

def normalize_matrix(matrix):
    """
    Normalize using robust statistics: clip extremes at 1st and 99th percentile then scale to [0,1].
    """
    lower, upper = np.percentile(matrix, [1, 99])
    matrix_clipped = np.clip(matrix, lower, upper)
    norm_matrix = (matrix_clipped - lower) / (upper - lower + 1e-8)
    return norm_matrix

def save_image(matrix, output_path):
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
