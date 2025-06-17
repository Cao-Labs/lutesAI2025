import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Load model
print("Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read sequences one at a time
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

# Batch and process sequences
batch_size = 8
batch = []
ids = []

def process_batch(batch, ids):
    for seq, seq_id in zip(batch, ids):
        try:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)

            with torch.no_grad():
                emb = model.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(return_per_residue_embeddings=True)
                ).per_residue_embedding.detach().cpu()

            # Clean and round
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            emb = torch.round(emb * 1000) / 1000

            # Save
            torch.save(emb, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        except Exception as e:
            print(f"Error processing {seq_id}: {e}")

# Main loop
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or len(seq) >= 30000:
        print(f"Skipping {seq_id} (invalid or too long)")
        continue

    batch.append(seq)
    ids.append(seq_id)

    if len(batch) == batch_size:
        process_batch(batch, ids)
        print(f"Processed {idx} sequences...")
        batch, ids = [], []

if batch:
    process_batch(batch, ids)

print(f"Done. All embeddings saved to: {OUTPUT_DIR}")

