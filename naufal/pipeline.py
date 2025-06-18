import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
FEATURES_FILE = "/data/summer2020/naufal/protein_features.txt"
OUTPUT_DIR = "/data/summer2020/naufal/final_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

L_FIXED = 4096
D_ORIG = 1536
D_FINAL = D_ORIG + 4 + 1
BATCH_SIZE = 1000
NUM_WORKERS = 4  # Tune based on your CPU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-hot encoding for SS
ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: torch.eye(len(ss_vocab))[i] for i, ch in enumerate(ss_vocab)}

# Load SS/RSA features into shared dictionary
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

# Define the processing function
def process_protein(fname):
    try:
        prot_id = fname[:-3]
        fpath = os.path.join(EMBEDDINGS_DIR, fname)

        if prot_id not in features:
            return (prot_id, "missing_features")

        # Load embedding
        embedding = torch.load(fpath, map_location="cpu")
        L = embedding.shape[0]

        ss_tensor, rsa_tensor = features[prot_id]
        if ss_tensor.shape[0] != L or rsa_tensor.shape[0] != L:
            return (prot_id, "length_mismatch")

        # Move to GPU
        embedding = embedding.to(DEVICE)
        ss_tensor = ss_tensor.to(DEVICE)
        rsa_tensor = rsa_tensor.to(DEVICE)

        # Concatenate
        full_tensor = torch.cat([embedding, ss_tensor, rsa_tensor], dim=1)

        # Fix length
        L_current = full_tensor.shape[0]
        if L_current < L_FIXED:
            pad = torch.zeros((L_FIXED - L_current, D_FINAL), dtype=full_tensor.dtype, device=DEVICE)
            full_tensor = torch.cat([full_tensor, pad], dim=0)
        elif L_current > L_FIXED:
            full_tensor = full_tensor[:L_FIXED, :]

        # Normalize (min-max)
        min_vals = full_tensor.min(dim=0, keepdim=True).values
        max_vals = full_tensor.max(dim=0, keepdim=True).values
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0
        full_tensor = (full_tensor - min_vals) / diff

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{prot_id}.pt")
        torch.save(full_tensor, out_path)
        return (prot_id, "success")

    except Exception as e:
        return (fname[:-3], f"error: {str(e)}")

# Main loop
all_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".pt")]
total_files = len(all_files)
success, skipped = 0, 0

print(f"Starting batch processing with {NUM_WORKERS} workers...\n")

with torch.no_grad():
    for i in range(0, total_files, BATCH_SIZE):
        batch = all_files[i:i + BATCH_SIZE]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_protein, fname) for fname in batch]
            for future in as_completed(futures):
                prot_id, status = future.result()
                if status == "success":
                    success += 1
                    if success == 1 or success % 50000 == 0:
                        print(f"Processed {success} proteins...")
                else:
                    skipped += 1
                    print(f"Skipped {prot_id}: {status}")

# Summary
print("\nFinished processing.")
print(f"Total successfully processed: {success}")
print(f"Total skipped: {skipped}")



