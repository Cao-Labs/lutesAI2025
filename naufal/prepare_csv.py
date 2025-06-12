import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Config
DATA_DIR = "/data/summer2020/naufal"
EMBEDDING_DIR = os.path.join(DATA_DIR, "esm3_embeddings")
FEATURE_FILE = os.path.join(DATA_DIR, "protein_features.txt")
CSV_OUTPUT = os.path.join(DATA_DIR, "merged_features.csv")
LOG_FILE = os.path.join(DATA_DIR, "skipped_proteins.log")
MAX_LEN = 1024

# Step 1: Parse secondary structure file
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

# Step 2: Combine embeddings + secondary structure and save to CSV
print("Merging and writing to CSV...")
records = []
skipped_logs = []

for filename in os.listdir(EMBEDDING_DIR):
    if not filename.endswith(".pt"):
        continue
    pid = filename.replace(".pt", "")
    emb_path = os.path.join(EMBEDDING_DIR, filename)
    if pid not in feature_map:
        skipped_logs.append(f"{pid} skipped: no structure found")
        continue

    embedding = torch.load(emb_path)

    # Ensure 2D embedding
    if embedding.dim() == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    elif embedding.dim() == 3:
        skipped_logs.append(f"{pid} skipped: unexpected embedding shape {embedding.shape}")
        continue

    structure = torch.tensor(feature_map[pid])

    if structure.dim() == 3 and structure.shape[0] == 1:
        structure = structure.squeeze(0)
    elif structure.dim() == 3:
        skipped_logs.append(f"{pid} skipped: unexpected structure shape {structure.shape}")
        continue

    if embedding.shape[0] != structure.shape[0]:
        skipped_logs.append(f"{pid} skipped: length mismatch {embedding.shape[0]} vs {structure.shape[0]}")
        continue

    combined = torch.cat([embedding, structure], dim=1)
    record = {
        "ProteinID": pid,
        "Features": json.dumps(combined.tolist())
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved merged CSV: {CSV_OUTPUT}")

# Save skipped logs
with open(LOG_FILE, "w") as logf:
    for line in skipped_logs:
        logf.write(line + "\n")
print(f"Skipped proteins logged to: {LOG_FILE}")

# Step 3: Dataset class for BigBird
class ProteinDataset(Dataset):
    def __init__(self, csv_path, max_len=1024):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid = self.data.iloc[idx]["ProteinID"]
        features = json.loads(self.data.iloc[idx]["Features"])
        x = torch.tensor(features, dtype=torch.float32)

        length = x.shape[0]
        if length > self.max_len:
            x = x[:self.max_len]
            mask = torch.ones(self.max_len)
        else:
            pad_len = self.max_len - length
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = torch.cat([torch.ones(length), torch.zeros(pad_len)])

        return x, mask, pid

# Step 4: Example usage
def create_dataloader(csv_file, batch_size=4):
    dataset = ProteinDataset(csv_file, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Uncomment to test the dataloader
# loader = create_dataloader(CSV_OUTPUT)
# for x, mask, pid in loader:
#     print(f"Batch shape: {x.shape}, Mask shape: {mask.shape}")
#     break
