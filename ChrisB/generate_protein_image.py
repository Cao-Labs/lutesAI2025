import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import argparse
from Bio import SeqIO
# headers 
#python generate_protein_image.py --sequence "MKTFFVLLLCTFTVSGTANAQDNPKTITISNDGTY" --out test_protein.png
def load_esm3_model():
    print("Loading ESM-3 model...")
    model, alphabet = esm.pretrained.esm3_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    return model, batch_converter

def get_embeddings(model, batch_converter, sequence, sequence_id="protein"):
    batch_labels, batch_strs, batch_tokens = batch_converter([(sequence_id, sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36])
    token_embeddings = results["representations"][36]
    return token_embeddings[0, 1:-1]  # Remove [CLS] and [EOS]

def to_2d_matrix(embedding_matrix, method="reshape"):
    seq_len, embed_dim = embedding_matrix.size()
    
    if method == "reshape":
        # Pad to square then reshape
        size = int(np.ceil(np.sqrt(embed_dim)))
        padded = torch.zeros((seq_len, size * size))
        padded[:, :embed_dim] = embedding_matrix
        matrix = padded.view(seq_len, size, size).mean(dim=0).numpy()
    
    elif method == "meanpool":
        matrix = embedding_matrix.mean(dim=0).view(int(np.sqrt(embed_dim)), -1).numpy()

    else:
        raise ValueError("Unsupported method. Use 'reshape' or 'meanpool'.")
    
    return matrix

def save_image(matrix, out_path="protein_image.png"):
    plt.imshow(matrix, cmap='viridis')
    plt.title("Protein Feature Image (ESM-3)")
    plt.colorbar()
    plt.savefig(out_path)
    print(f"Image saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, help="Protein sequence (in quotes)", required=True)
    parser.add_argument("--out", type=str, default="protein_image.png", help="Output image path")
    args = parser.parse_args()

    model, batch_converter = load_esm3_model()
    embedding_matrix = get_embeddings(model, batch_converter, args.sequence)
    matrix = to_2d_matrix(embedding_matrix)
    save_image(matrix, args.out)

if __name__ == "__main__":
    main()
