import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.pairwise import cosine_similarity
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
    return embedding  # shape: (seq_len, embed_dim)

def to_similarity_matrix(embedding):
    # Convert to numpy and compute cosine similarity between residues
    embedding_np = embedding.cpu().numpy()
    sim_matrix = cosine_similarity(embedding_np)
    return sim_matrix  # shape: (seq_len, seq_len)

def normalize_matrix(matrix):
    # Normalize to [0, 1] range for visualization
    min_val, max_val = matrix.min(), matrix.max()
    return (matrix - min_val) / (max_val - min_val + 1e-8)

def save_image(matrix, output_path):
    normalized = normalize_matrix(matrix)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(normalized, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.title("Protein Feature Image (ESM-3 Cosine Similarity)")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate protein feature similarity image from sequence.")
    parser.add_argument("--sequence", type=str, required=True, help="Protein sequence in quotes")
    parser.add_argument("--out", type=str, default="protein_image.png", help="Output image filename")
    args = parser.parse_args()

    model = load_esm3_model()
    embedding = generate_embedding(model, args.sequence)
    matrix = to_similarity_matrix(embedding)
    save_image(matrix, args.out)

if __name__ == "__main__":
    main()
