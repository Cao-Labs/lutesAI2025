import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from esm.utils.constants.models import ESM3_OPEN_SMALL

# === Config ===
FINAL_EMBEDDINGS_DIR = "/data/archives/naufal/final_embeddings"
SEQUENCE_SOURCE = "/data/summer2020/naufal/training_data/sequence_dict.pt"  # Assuming you have a preloaded sequence mapping
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load ESM-3 model ===
print("[INFO] Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)

# === Load sequences from dict (ID -> sequence) ===
print("[INFO] Loading sequence dictionary...")
sequence_dict = torch.load(SEQUENCE_SOURCE)  # Assumed to be a dict {id: seq}

# === Get only already-processed protein IDs ===
print("[INFO] Scanning final_embeddings/ for protein IDs...")
protein_ids = {
    fname[:-3] for fname in os.listdir(FINAL_EMBEDDINGS_DIR) if fname.endswith(".pt")
}
print(f"[INFO] Found {len(protein_ids)} protein IDs to embed.")

# === Process embeddings ===
processed = 0
skipped = 0

for pid in sorted(protein_ids):
    out_path = os.path.join(OUTPUT_DIR, f"{pid}.pt")
    if os.path.exists(out_path):
        continue  # Already done

    if pid not in sequence_dict:
        print(f"[SKIP] No sequence for {pid}")
        skipped += 1
        continue

    seq = sequence_dict[pid]
    if not seq or len(seq) >= 30000:
        print(f"[SKIP] Invalid or too long: {pid}")
        skipped += 1
        continue

    try:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)

        with torch.no_grad():
            output = model(sequence_tokens=protein_tensor.sequence[None]).embeddings.detach().cpu()[0]
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            output = torch.round(output * 1000) / 1000

        torch.save(output, out_path)
        processed += 1

        if processed % 100 == 0:
            print(f"[✓] {processed} proteins embedded...")

    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        skipped += 1

print(f"\nDone: {processed} embedded | {skipped} skipped")
