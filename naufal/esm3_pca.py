import os
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

# CONFIGURATION
INPUT_DIR = "/data/summer2020/naufal/esm3_embeddings"
OUTPUT_FILE = "/data/summer2020/naufal/esm3_embeddings_pca2_padded.pt"
SEQ_LENGTH = 512
REDUCED_DIM = 2
BATCH_SIZE = 5000  # Number of residues per PCA batch

# SAFETY CHECK: ensure tensor is 2D
def ensure_2d(embedding_tensor, filename):
    if embedding_tensor.ndim == 3:
        if embedding_tensor.shape[-1] == 1:
            embedding_tensor = embedding_tensor.squeeze(-1)
        else:
            raise ValueError(f"File {filename} has shape {embedding_tensor.shape}, expected last dim to be 1.")
    elif embedding_tensor.ndim != 2:
        raise ValueError(f"File {filename} has unexpected shape {embedding_tensor.shape}, expected 2D.")
    return embedding_tensor

# STEP 1: Gather all .pt files
embedding_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")]
print(f"Found {len(embedding_files)} files.")

# STEP 2: Initialize Incremental PCA
ipca = IncrementalPCA(n_components=REDUCED_DIM, batch_size=BATCH_SIZE)
batch = []
batch_count = 0

# STEP 3: First pass — fit Incremental PCA in batches
print("Starting PCA fitting...")
for idx, fname in enumerate(embedding_files):
    emb = torch.load(os.path.join(INPUT_DIR, fname))
    emb = ensure_2d(emb, fname)
    batch.append(emb.numpy())

    if sum(b.shape[0] for b in batch) >= BATCH_SIZE:
        combined = np.vstack(batch)
        ipca.partial_fit(combined)
        batch_count += 1
        print(f"PCA batch {batch_count} fitted with {combined.shape[0]} residues")
        batch = []

# Fit remaining batch if any
if batch:
    combined = np.vstack(batch)
    ipca.partial_fit(combined)
    batch_count += 1
    print(f"PCA batch {batch_count} fitted with {combined.shape[0]} residues")

print("PCA fitting complete.")

# STEP 4: Transform each file, pad/truncate to fixed length
processed = {}
print("Transforming and saving embeddings...")
for idx, fname in enumerate(embedding_files):
    emb = torch.load(os.path.join(INPUT_DIR, fname))
    emb = ensure_2d(emb, fname)
    reduced = ipca.transform(emb.numpy())  # shape: [L, 2]

    # Truncate or pad to SEQ_LENGTH
    L = reduced.shape[0]
    if L >= SEQ_LENGTH:
        final = reduced[:SEQ_LENGTH]
    else:
        pad = np.zeros((SEQ_LENGTH - L, REDUCED_DIM))
        final = np.vstack([reduced, pad])

    processed[fname] = torch.tensor(final, dtype=torch.float32)

    if (idx + 1) % 500 == 0 or idx == len(embedding_files) - 1:
        print(f"Processed {idx + 1} / {len(embedding_files)} files")

# STEP 5: Save all to a single file
torch.save(processed, OUTPUT_FILE)
print(f"Done. Embeddings saved to {OUTPUT_FILE}")


