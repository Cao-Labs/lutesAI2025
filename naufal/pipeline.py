import os
import torch

# === Config ===
EMBEDDINGS_DIR = "/data/summer2020/naufal/testing_embeddings"
FEATURES_FILE = "/data/summer2020/naufal/testing_features.txt"
OUTPUT_DIR = "/data/summer2020/naufal/testing_normalized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

L_FIXED = 1913
D_ORIG = 1536
D_FINAL = D_ORIG + 4 + 1  # 4 for SS one-hot, 1 for RSA

ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: torch.eye(len(ss_vocab))[i] for i, ch in enumerate(ss_vocab)}

# === Step 1: Load and index features file ===
print("[INFO] Loading and indexing features file...")
features_dict = {}
with open(FEATURES_FILE, "r") as f:
    current_id = None
    ss_list = []
    rsa_list = []
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            if current_id and ss_list and rsa_list:
                features_dict[current_id] = (
                    torch.stack(ss_list),
                    torch.tensor(rsa_list).unsqueeze(1)
                )
            current_id = line[1:].strip()
            ss_list = []
            rsa_list = []
        else:
            try:
                ss, rsa = line.split("\t")
                ss_tensor = ss_to_onehot.get(ss, ss_to_onehot['L'])
                ss_list.append(ss_tensor)
                rsa_list.append(float(rsa))
            except:
                continue
    # Save last block
    if current_id and ss_list and rsa_list:
        features_dict[current_id] = (
            torch.stack(ss_list),
            torch.tensor(rsa_list).unsqueeze(1)
        )

print(f"[INFO] Loaded {len(features_dict)} protein features.")

# === Step 2: Process embeddings ===
processed = 0
skipped = 0

for fname in os.listdir(EMBEDDINGS_DIR):
    if not fname.endswith(".pt"):
        continue

    prot_id = fname[:-3]
    pt_path = os.path.join(EMBEDDINGS_DIR, fname)

    try:
        embedding = torch.load(pt_path)
        embedding = embedding[1:-1, :]  # Drop first and last

        if prot_id not in features_dict:
            print(f"[Skipped] {prot_id}: No features found")
            skipped += 1
            continue

        ss_tensor, rsa_tensor = features_dict[prot_id]

        if ss_tensor.shape[0] != embedding.shape[0]:
            print(f"[Skipped] {prot_id}: Length mismatch")
            skipped += 1
            continue

        combined = torch.cat([embedding, ss_tensor, rsa_tensor], dim=1)

        # Pad or trim
        if combined.shape[0] < L_FIXED:
            pad = torch.zeros((L_FIXED - combined.shape[0], D_FINAL))
            combined = torch.cat([combined, pad], dim=0)
        elif combined.shape[0] > L_FIXED:
            combined = combined[:L_FIXED, :]

        # Normalize
        min_vals = combined.min(dim=0, keepdim=True).values
        max_vals = combined.max(dim=0, keepdim=True).values
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0
        combined = (combined - min_vals) / diff

        torch.save(combined, os.path.join(OUTPUT_DIR, f"{prot_id}.pt"))
        processed += 1

        if processed == 1:
            print(f"[✓] First protein written: {prot_id}")
        elif processed % 10000 == 0:
            print(f"[✓] {processed} proteins written...")

    except Exception as e:
        skipped += 1
        print(f"[Error] Skipped {prot_id}: {str(e)}")
        continue

print(f"\nDONE: {processed} processed | {skipped} skipped")




