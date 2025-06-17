import os
import torch
import numpy as np

# Paths
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
FEATURES_FILE = "/data/summer2020/naufal/protein_features.txt"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# One-hot encoding for secondary structure
ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: np.eye(len(ss_vocab))[i] for i, ch in enumerate(ss_vocab)}

# Step 1: Load features from file
features = {}
with open(FEATURES_FILE, "r") as f:
    current_id = None
    ss_rows = []
    rsa_rows = []

    for line in f:
        line = line.strip()
        if line.startswith("#"):
            if current_id and ss_rows and rsa_rows:
                features[current_id] = (ss_rows, rsa_rows)
            current_id = line[1:].strip()
            ss_rows, rsa_rows = [], []
        else:
            try:
                ss_char, rsa_str = line.split("\t")
                onehot = ss_to_onehot.get(ss_char, ss_to_onehot['L'])  # Default to 'L'
                rsa_val = float(rsa_str)
                ss_rows.append(onehot)
                rsa_rows.append([rsa_val])
            except ValueError:
                continue

    if current_id and ss_rows and rsa_rows:
        features[current_id] = (ss_rows, rsa_rows)

# Step 2–4: Merge features with embeddings
print("Augmenting embeddings with SS and RSA (no normalization)...")

success, skipped = 0, 0

for filename in os.listdir(EMBEDDINGS_DIR):
    if not filename.endswith(".pt"):
        continue

    protein_id = filename[:-3]
    embedding_path = os.path.join(EMBEDDINGS_DIR, filename)

    try:
        embedding = torch.load(embedding_path)
        L, D = embedding.shape

        if protein_id not in features:
            skipped += 1
            continue

        ss_array, rsa_array = features[protein_id]

        if len(ss_array) != L:
            skipped += 1
            continue

        ss_tensor = torch.tensor(np.array(ss_array), dtype=torch.float32)
        rsa_tensor = torch.tensor(np.array(rsa_array), dtype=torch.float32)

        augmented = torch.cat([embedding, ss_tensor, rsa_tensor], dim=1)
        torch.save(augmented, os.path.join(OUTPUT_DIR, f"{protein_id}.pt"))
        success += 1

        if success == 1 or success % 50000 == 0:
            print(f"Processed {success} proteins...")

    except Exception as e:
        print(f"Error processing {protein_id}: {e}")
        skipped += 1

# Summary
print(f"\nTotal successfully processed: {success}")
print(f"Total skipped: {skipped}")
