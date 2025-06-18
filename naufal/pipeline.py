import os
import torch
import numpy as np

# Paths
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
FEATURES_FILE = "/data/summer2020/naufal/protein_features.txt"
OUTPUT_DIR = "/data/summer2020/naufal/final_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
L_FIXED = 4096
D_ORIG = 1536
D_FINAL = D_ORIG + 4 + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-hot encoding for SS
ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: torch.eye(len(ss_vocab))[i] for i, ch in enumerate(ss_vocab)}

# Load SS and RSA features
print("Loading SS and RSA features...")
features = {}
with open(FEATURES_FILE, "r") as f:
    current_id = None
    ss_rows, rsa_rows = [], []

    for line in f:
        line = line.strip()
        if line.startswith("#"):
            if current_id and ss_rows and rsa_rows:
                ss_tensor = torch.stack(ss_rows)
                rsa_tensor = torch.tensor(rsa_rows).unsqueeze(1)
                features[current_id] = (ss_tensor, rsa_tensor)
            current_id = line[1:].strip()
            ss_rows, rsa_rows = [], []
        else:
            try:
                ss_char, rsa_val = line.split("\t")
                onehot = ss_to_onehot.get(ss_char, ss_to_onehot['L'])
                ss_rows.append(onehot)
                rsa_rows.append(float(rsa_val))
            except:
                continue

    if current_id and ss_rows and rsa_rows:
        ss_tensor = torch.stack(ss_rows)
        rsa_tensor = torch.tensor(rsa_rows).unsqueeze(1)
        features[current_id] = (ss_tensor, rsa_tensor)

print(f"Loaded SS/RSA features for {len(features)} proteins.\n")

# Fix sequence length to L = 4096
def fix_length(tensor, L_fixed=4096):
    L, D = tensor.shape
    if L == L_fixed:
        return tensor
    elif L > L_fixed:
        return tensor[:L_fixed, :]
    else:
        pad = torch.zeros((L_fixed - L, D), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=0)

# Min-Max normalization (per tensor)
def normalize_minmax(tensor):
    min_vals = tensor.min(dim=0, keepdim=True).values
    max_vals = tensor.max(dim=0, keepdim=True).values
    diff = max_vals - min_vals
    diff[diff == 0] = 1.0  # prevent divide-by-zero
    return (tensor - min_vals) / diff

# Main loop
success, skipped = 0, 0
print("Processing and writing training-ready embeddings...\n")

for fname in os.listdir(EMBEDDINGS_DIR):
    if not fname.endswith(".pt"):
        continue

    prot_id = fname[:-3]
    fpath = os.path.join(EMBEDDINGS_DIR, fname)

    try:
        # Load embedding
        embedding = torch.load(fpath, map_location="cpu")  # [L, 1536]
        L = embedding.shape[0]

        # Get matching SS/RSA
        if prot_id not in features:
            skipped += 1
            continue

        ss_tensor, rsa_tensor = features[prot_id]
        if ss_tensor.shape[0] != L or rsa_tensor.shape[0] != L:
            skipped += 1
            continue

        # Move to GPU
        embedding = embedding.to(DEVICE)
        ss_tensor = ss_tensor.to(DEVICE)
        rsa_tensor = rsa_tensor.to(DEVICE)

        # Combine all features: [L, 1541]
        full_tensor = torch.cat([embedding, ss_tensor, rsa_tensor], dim=1)

        # Fix sequence length to 4096
        full_tensor = fix_length(full_tensor, L_fixed=L_FIXED)

        # Normalize with Min-Max
        full_tensor = normalize_minmax(full_tensor)

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{prot_id}.pt")
        torch.save(full_tensor, out_path)

        success += 1
        if success == 1 or success % 50000 == 0:
            print(f"Processed {success} proteins...")

    except Exception as e:
        print(f"Error with {prot_id}: {e}")
        skipped += 1

# Summary
print("\nAll done.")
print(f"Total successfully processed: {success}")
print(f"Total skipped: {skipped}")

