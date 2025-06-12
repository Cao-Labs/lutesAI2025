import os
import json
import pandas as pd
import torch
import torch.nn.functional as F

# Config
DATA_DIR = "/data/summer2020/naufal"
EMBEDDING_DIR = os.path.join(DATA_DIR, "esm3_embeddings")
FEATURE_FILE = os.path.join(DATA_DIR, "protein_features.txt")
CSV_OUTPUT = os.path.join(DATA_DIR, "esm3_fixed_embeddings.csv")
MAX_LEN = 1024

# Step 1: Load secondary structure
print("Loading secondary structure...")
feature_map = {}
current_id = None
current_features = []

with open(FEATURE_FILE) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            if current_id and current_features:
                feature_map[current_id] = current_features
            current_id = line[1:].strip()
            current_features = []
        elif line:
            parts = line.split()
            current_features.append([float(p) for p in parts[1:]])
if current_id and current_features:
    feature_map[current_id] = current_features

# Step 2: Fix embeddings and save to CSV
print("Fixing and saving embeddings to CSV...")
records = []
skipped = []

for filename in os.listdir(EMBEDDING_DIR):
    if not filename.endswith(".pt"):
        continue
    pid = filename.replace(".pt", "")
    emb_path = os.path.join(EMBEDDING_DIR, filename)
    if pid not in feature_map:
        skipped.append(f"{pid} skipped: no structure found")
        continue

    embedding = torch.load(emb_path)

    # Squeeze batch dim if needed
    if embedding.dim() == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    elif embedding.dim() == 3:
        skipped.append(f"{pid} skipped: unexpected embedding shape {embedding.shape}")
        continue

    structure = torch.tensor(feature_map[pid])

    if structure.dim() == 3 and structure.shape[0] == 1:
        structure = structure.squeeze(0)
    elif structure.dim() == 3:
        skipped.append(f"{pid} skipped: unexpected structure shape {structure.shape}")
        continue

    if embedding.shape[0] != structure.shape[0]:
        skipped.append(f"{pid} skipped: length mismatch {embedding.shape[0]} vs {structure.shape[0]}")
        continue

    combined = torch.cat([embedding, structure], dim=1)

    # Pad or truncate
    length = combined.shape[0]
    if length > MAX_LEN:
        combined = combined[:MAX_LEN]
    else:
        pad_len = MAX_LEN - length
        combined = F.pad(combined, (0, 0, 0, pad_len))  # pad rows

    records.append({
        "ProteinID": pid,
        "Features": json.dumps(combined.tolist())
    })

# Save CSV
df = pd.DataFrame(records)
df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved fixed embeddings CSV to: {CSV_OUTPUT}")

# Save skipped
log_path = os.path.join(DATA_DIR, "esm3_skipped.log")
with open(log_path, "w") as logf:
    for line in skipped:
        logf.write(line + "\n")
print(f"Skipped proteins logged to: {log_path}")
