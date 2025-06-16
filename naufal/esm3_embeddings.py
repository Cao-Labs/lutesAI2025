import os
import torch
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

# Authenticate (only needed once if you've already done huggingface-cli login)
login()

# Load model
print("Loading ESM-3 model...")
model = ESM3.from_pretrained("esm3-open").to("cuda")  # or "cpu"

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_tryforward"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FASTA reader
def fasta_reader(path):
    with open(path, "r") as f:
        identifier, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if identifier:
                    yield identifier, "".join(seq)
                identifier = line[1:]
                seq = []
            else:
                seq.append(line)
        if identifier:
            yield identifier, "".join(seq)

# Try inference without generation
print("Trying direct embedding inference without generation...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    try:
        if not seq or set(seq) == {"."}:
            print(f"Skipping {seq_id} (invalid)")
            continue

        protein = ESMProtein(sequence=seq)

        # Attempt forward-like inference
        output = model.infer(protein)

        if hasattr(protein, "error"):
            print(f"Error from model for {seq_id}: {protein.error}")
            continue

        if not hasattr(protein, "representations") or "sequence" not in protein.representations:
            print(f"No sequence embedding found for {seq_id}, skipping.")
            continue

        emb = protein.representations["sequence"]
        emb_tensor = torch.tensor(emb)
        emb_tensor = torch.nan_to_num(emb_tensor, nan=0.0)
        emb_tensor[torch.isinf(emb_tensor)] = 0.0
        emb_tensor = torch.round(emb_tensor * 1000) / 1000

        torch.save(emb_tensor, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 50 == 0:
            print(f"{idx} sequences processed...")

    except Exception as e:
        print(f"Failed {seq_id}: {e}")

print(f"Done. Saved to: {OUTPUT_DIR}")



