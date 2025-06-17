import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import argparse
from esm.pretrained import load_model_and_alphabet_local

def load_esm3_model():
    model_path = "/data/summer2020/naufal/esm3_embeddings/KDSA_ACIAD.pt"  # Your local ESM-3 model file
    print(f"Loading ESM-3 model from {model_path}...")
    model, alphabet = load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, batch_converter

def get_embeddings(model, batch_converter, sequence, sequence_id="protein"):
    batch_labels, batch_strs, batch_tokens = batch_converter([(sequence_id, sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers])
    token_embeddings = results["representations"][model.num_layers]
    return token_embeddings[0, 1:-1]  # Exclude [CLS] and [EOS]

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

    model, batch_converter = load_esm3_model()
    embedding = get_embeddings(model, batch_converter, args.sequence)
    matrix = to_2d_matrix(embedding)
    save_image(matrix, args.out)

if __name__ == "__main__":
    main()
