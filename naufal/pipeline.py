import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Paths ===
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
FEATURES_FILE = "/data/summer2020/naufal/features_dssp_direct.txt"
OUTPUT_DIR = "/data/summer2020/naufal/final_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Constants ===
L_FIXED = 1895
D_ORIG = 1536
D_FINAL = D_ORIG + 4 + 1
BATCH_SIZE = 1000
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === SS One-hot encoding ===
ss_vocab = ['H', 'E', 'C', 'L']
ss_to_onehot = {ch: torch.eye(len(ss_vocab))[i] for i, ch in enumerate(ss_vocab)}

def load_features(protein_id):
    ss_rows, rsa_rows = [], []
    found = False
    with open(FEATURES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if found:
                    break
                found = (line[1:].strip() == protein_id)
                continue
            if found:
                try:
                    ss, rsa = line.split("\t")
                    ss_rows.append(ss_to_onehot.get(ss, ss_to_onehot['L']))
                    rsa_rows.append(float(rsa))
                except:
                    continue
    if not ss_rows or not rsa_rows:
        return None, None
    return torch.stack(ss_rows), torch.tensor(rsa_rows).unsqueeze(1)

def process_protein(fname):
    try:
        prot_id = fname[:-3]
        fpath = os.path.join(EMBEDDINGS_DIR, fname)
        embedding = torch.load(fpath, map_location="cpu")[1:-1]  # Drop <CLS> and <EOS>
        L = embedding.shape[0]

        ss, rsa = load_features(prot_id)
        if ss is None or rsa is None:
            return prot_id, "missing_features"
        if ss.shape[0] != L:
            return prot_id, f"length_mismatch (L={L}, SS={ss.shape[0]})"

        embedding = embedding.to(DEVICE)
        ss = ss.to(DEVICE)
        rsa = rsa.to(DEVICE)

        combined = torch.cat([embedding, ss, rsa], dim=1)

        # Pad/truncate to L_FIXED
        if combined.shape[0] < L_FIXED:
            padding = torch.zeros((L_FIXED - combined.shape[0], D_FINAL), device=DEVICE)
            combined = torch.cat([combined, padding], dim=0)
        elif combined.shape[0] > L_FIXED:
            combined = combined[:L_FIXED, :]

        # Normalize (min-max)
        min_vals = combined.min(dim=0, keepdim=True).values
        max_vals = combined.max(dim=0, keepdim=True).values
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0
        combined = (combined - min_vals) / diff

        # Save
        torch.save(combined, os.path.join(OUTPUT_DIR, f"{prot_id}.pt"))
        return prot_id, "success"

    except Exception as e:
        return fname[:-3], f"error: {str(e)}"

# === Main Loop ===
all_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".pt")]
success = skipped = 0

print(f"Processing {len(all_files)} proteins...")

with torch.no_grad():
    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i:i + BATCH_SIZE]
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_protein, fname) for fname in batch]
            for future in as_completed(futures):
                pid, status = future.result()
                if status == "success":
                    success += 1
                    if success == 1 or success % 50000 == 0:
                        print(f"Processed {success} proteins...")
                else:
                    skipped += 1
                    print(f"Skipped {pid}: {status}")

print(f"\nAll Done! Processed: {success}, Skipped: {skipped}")




