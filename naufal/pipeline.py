import os
import torch

# === Config ===
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
FEATURES_FILE = "/data/summer2020/naufal/features_dssp_direct.txt"
OUTPUT_DIR = "/data/summer2020/naufal/final_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

L_FIXED = 1895
D_ORIG = 1536
D_FINAL = D_ORIG + 4 + 1  # 4 for SS one-hot, 1 for RSA

ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: torch.eye(len(ss_vocab))[i] for i, ch in enumerate(ss_vocab)}

# === Get features for a given ID ===
def get_features(prot_id):
    with open(FEATURES_FILE, "r") as f:
        found = False
        ss_list = []
        rsa_list = []
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if found:
                    break
                found = line[1:].strip() == prot_id
            elif found:
                try:
                    ss, rsa = line.split("\t")
                    ss_list.append(ss_to_onehot.get(ss, ss_to_onehot['L']))
                    rsa_list.append(float(rsa))
                except:
                    continue
        if found and ss_list and rsa_list:
            return torch.stack(ss_list), torch.tensor(rsa_list).unsqueeze(1)
        return None, None

# === Process each embedding file ===
processed = 0
skipped = 0

for fname in os.listdir(EMBEDDINGS_DIR):
    if not fname.endswith(".pt"):
        continue

    prot_id = fname[:-3]
    pt_path = os.path.join(EMBEDDINGS_DIR, fname)

    try:
        print(f"[Loading] {prot_id}...")
        embedding = torch.load(pt_path)
        print(f"[Loaded] {prot_id} | Shape: {embedding.shape}")

        embedding = embedding[1:-1, :]  # Drop first and last
        print(f"[Trimmed] {prot_id} | New shape: {embedding.shape}")

        ss_tensor, rsa_tensor = get_features(prot_id)
        if ss_tensor is None or rsa_tensor is None:
            print(f"[Skipped] {prot_id}: Missing SS/RSA features")
            skipped += 1
            continue

        if ss_tensor.shape[0] != embedding.shape[0]:
            print(f"[Skipped] {prot_id}: Length mismatch (SS/RSA vs Embedding)")
            skipped += 1
            continue

        print(f"[Appending] {prot_id}")
        combined = torch.cat([embedding, ss_tensor, rsa_tensor], dim=1)

        if combined.shape[0] < L_FIXED:
            print(f"[Padding] {prot_id}")
            pad = torch.zeros((L_FIXED - combined.shape[0], D_FINAL))
            combined = torch.cat([combined, pad], dim=0)
        elif combined.shape[0] > L_FIXED:
            print(f"[Truncating] {prot_id}")
            combined = combined[:L_FIXED, :]

        print(f"[Normalizing] {prot_id}")
        min_vals = combined.min(dim=0, keepdim=True).values
        max_vals = combined.max(dim=0, keepdim=True).values
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0
        combined = (combined - min_vals) / diff

        torch.save(combined, os.path.join(OUTPUT_DIR, f"{prot_id}.pt"))
        processed += 1

        if processed == 1 or processed % 10000 == 0:
            print(f"[Progress] Processed {processed} files...")

    except Exception as e:
        skipped += 1
        print(f"[Error] Skipped {prot_id}: {str(e)}")
        continue

print(f"\n🎉 DONE: {processed} processed | {skipped} skipped")



