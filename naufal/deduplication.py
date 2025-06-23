import os
import torch
import hashlib

# === Config ===
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
FEATURES_FILE = "/data/summer2020/naufal/features_dssp_direct.txt"
FEATURES_OUTPUT = "/data/summer2020/naufal/features_dssp_deduplicated.txt"

# === Step 1: Hash tensor contents and track duplicates ===
def tensor_hash(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

print("[INFO] Scanning embeddings for duplicate content...")

hash_to_id = {}
duplicate_ids = set()

for fname in os.listdir(EMBEDDINGS_DIR):
    if not fname.endswith(".pt"):
        continue

    prot_id = fname[:-3]
    fpath = os.path.join(EMBEDDINGS_DIR, fname)

    try:
        tensor = torch.load(fpath)
        h = tensor_hash(tensor)

        if h in hash_to_id:
            # Duplicate detected
            duplicate_ids.add(prot_id)
            os.remove(fpath)
            print(f"[Deleted] Duplicate embedding: {prot_id} (same as {hash_to_id[h]})")
        else:
            hash_to_id[h] = prot_id
    except Exception as e:
        print(f"[Error] Could not read {fname}: {e}")
        continue

print(f"[INFO] Removed {len(duplicate_ids)} duplicate .pt files.")

# === Step 2: Filter features_dssp_direct.txt ===
print("[INFO] Cleaning features_dssp_direct.txt...")

with open(FEATURES_FILE, "r") as f:
    lines = f.readlines()

deduplicated_lines = []
write_block = False
current_id = None

for line in lines:
    stripped = line.strip()
    if stripped.startswith("#"):
        current_id = stripped[1:]
        write_block = current_id not in duplicate_ids
    if write_block:
        deduplicated_lines.append(line)

with open(FEATURES_OUTPUT, "w") as f:
    f.writelines(deduplicated_lines)

print(f"[DONE] Saved cleaned features to {FEATURES_OUTPUT}")

