import os
import numpy as np
from sklearn.decomposition import PCA
import torch

# CONFIG
INPUT_DIR = "/data/summer2020/naufal/esm3_embeddings"
OUTPUT_FILE = "/data/summer2020/naufal/esm3_embeddings_pca2_padded.pt"
SEQ_LENGTH = 512  # max length for BigBird input
REDUCED_DIM = 2   # PCA output dimension

# Collect all .pt embedding files
embedding_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")]
pca_data = []

# First pass: collect all embeddings for PCA fitting
print("Collecting data for PCA...")
for file in embedding_files:
    emb = torch.load(os.path.join(INPUT_DIR, file))
    if emb.ndim == 3:
        emb = emb.squeeze(-1)  # shape: [L, D]
    pca_data.append(emb.numpy())

all_embeddings = np.vstack(pca_data)
print("Fitting PCA...")
pca = PCA(n_components=REDUCED_DIM)
pca.fit(all_embeddings)

# Second pass: transform, pad/truncate, and store
processed = {}
print("Transforming and padding...")
for file in embedding_files:
    emb = torch.load(os.path.join(INPUT_DIR, file))
    if emb.ndim == 3:
        emb = emb.squeeze(-1)
    reduced = pca.transform(emb.numpy())  # shape: [L, 2]
    
    # Pad/truncate to fixed length
    L = reduced.shape[0]
    if L >= SEQ_LENGTH:
        final = reduced[:SEQ_LENGTH]
    else:
        pad = np.zeros((SEQ_LENGTH - L, REDUCED_DIM))
        final = np.vstack([reduced, pad])
    
    processed[file] = torch.tensor(final, dtype=torch.float32)

print("Saving output...")
torch.save(processed, OUTPUT_FILE)
print(f"Done. Saved to {OUTPUT_FILE}")
