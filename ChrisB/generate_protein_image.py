import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

def load_esm3_model():
    print("Loading ESM-3 model (multimodal, from Meta)...")
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(torch.float32)
    return model

def generate_embedding(model, sequence):
    protein = ESMProtein(sequence=sequence)
    with torch.no_grad():
        result = model.forward_and_sample(
            protein,
            SamplingConfig(return_per_residue_embeddings=True)
        )
        embedding = result.per_residue_embedding  # [L x D] tensor
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding

def to_2d_matrix(embedding_matrix):
    seq_len, embed_dim = embedding_matrix.size()
    size = int(np.ceil(np.sqrt(embed_dim)))
    padded = torch.zeros((seq_len, size * size))
    padded[:, :embed_dim] = embedding_matrix
    matrix = padded.view(seq_len, size, size).mean(dim=0).numpy()
    return matrix

def save_image(matrix, out_path):
    plt.imshow(matrix, cmap='viridis')
    plt.title("Protein Feature Image (ESM-3)")
    plt.colorbar()
    plt.savefig(out_path)
    print(f"Image saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, help="Protein sequence in quotes")
    parser.add_argument("--out", type=str, default="protein_image.png", help="Output image file name")
    args = parser.parse_args()

    model = load_esm3_model()
    embedding = generate_embedding(model, args.sequence)
    matrix = to_2d_matrix(embedding)
    save_image(matrix, args.out)

if __name__ == "__main__":
    main()
