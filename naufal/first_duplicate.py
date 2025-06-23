import os
from collections import defaultdict

# === Config ===
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings_new"

# === Scan filenames ===
seen = defaultdict(int)

for fname in os.listdir(EMBEDDINGS_DIR):
    if not fname.endswith(".pt"):
        continue
    prot_id = fname[:-3]  # remove .pt
    seen[prot_id] += 1
    if seen[prot_id] == 2:
        # Found the first duplicate
        count = sum(1 for f in os.listdir(EMBEDDINGS_DIR) if f.startswith(prot_id) and f.endswith(".pt"))
        print(f"[🚨 DUPLICATE FOUND] Protein ID: {prot_id}")
        print(f"Occurrences: {count}")
        break
else:
    print("No duplicates found.")
