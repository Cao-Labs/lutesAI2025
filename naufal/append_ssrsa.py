import os
import torch
import numpy as np

# Paths
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
FEATURES_FILE = "/data/summer2020/naufal/protein_features.txt"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# One-hot encoding for secondary structure (on CPU)
ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: torch.tensor(np.eye(len(ss_vocab))[i], dtype=torch.float32) for i, ch in enumerate(ss_vocab)}

# Step 1: Load features into memory-efficient dictionary (tensors)
features = {}
with open(FEATURES_FILE, "r") as f:
    current_id = None
    ss_rows = []
    rsa_rows = []

    for line in f:
        line = line.strip()
        if line.startswith("#"):
            if current_id and ss_rows and rsa_rows:
                ss_tensor = torch.stack(ss_rows)
                rsa_tensor = torch.tensor(rsa_rows, dtype=torch.float32).unsqueeze(1)
                features[current_id] = (ss_tensor, rsa_tensor)
            current_id = line[1:].strip()
            ss_rows, rsa_rows = [], []
        else:
            try:
                ss_char, rsa_str = line.split("\t")
                ss_onehot = ss_to_onehot.get(ss_char, ss_to_onehot['L'])  # default to 'L'
                rsa_val = float(rsa_str)
                ss_rows.append(ss_onehot)
                rsa_rows.append(rsa_val)
            except ValueError:
                continue

    if current_id and ss_rows and rsa_rows:
        ss_tensor = torch.stack(ss_rows)
        rsa_tensor = torch.tensor(rsa_rows, dtype=torch.float32).unsqueeze(1)
        features[current_id] = (ss_tensor, rsa_tensor)

# Step 2–4: Augment embeddings with SS and RSA
print("Augmenting embeddings with SS and RSA...")

success, skipped = 0, 0

for filename in os.listdir(EMBEDDINGS_DIR):
    if not filename.endswith(".pt"):
        continue

    protein_id = filename[:-3]
    embedding_path = os.path.join(EMBEDDINGS_DIR, filename)

    try:
        embedding = torch.load(embedding_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        L, D = embedding.shape

        if protein_id not in features:
            skipped += 1
            continue

        ss_tensor, rsa_tensor = features[protein_id]

        if ss_tensor.shape[0] != L:
            skipped += 1
            continue

        ss_tensor = ss_tensor.to(embedding.device)
        rsa_tensor = rsa_tensor.to(embedding.device)

        augmented = torch.cat((embedding, ss_tensor, rsa_tensor), dim=1)
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
