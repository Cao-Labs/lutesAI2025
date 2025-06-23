import os
import torch
import hashlib

# === Config ===
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings_new"

# === Function to hash a tensor's content ===
def tensor_hash(tensor):
    tensor_bytes = tensor.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

# === Track seen hashes ===
hash_to_id = {}

# === Process files ===
for fname in os.listdir(EMBEDDINGS_DIR):
    if not fname.endswith(".pt"):
        continue

    prot_id = fname[:-3]
    fpath = os.path.join(EMBEDDINGS_DIR, fname)

    try:
        tensor = torch.load(fpath)

        # Hash the tensor content
        h = tensor_hash(tensor)

        if h in hash_to_id:
            original = hash_to_id[h]
            print(f"[DUPLICATE EMBEDDING FOUND]")
            print(f"- First seen in: {original}")
            print(f"- Duplicate found in: {prot_id}")
            print(f"- SHA256 hash: {h}")
            break
        else:
            hash_to_id[h] = prot_id

    except Exception as e:
        print(f"[Error] Could not read {fname}: {e}")
        continue
else:
    print("No duplicate embeddings found.")

