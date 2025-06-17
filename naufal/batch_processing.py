import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Config
BATCH_SIZE = 8
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)

# Read FASTA
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

# Batch processor
def process_batch(batch_ids, batch_seqs):
    proteins = [ESMProtein(sequence=seq) for seq in batch_seqs]
    encoded = [model.encode(p).sequence for p in proteins]
    tokens = torch.stack(encoded).to("cuda")

    with torch.no_grad():
        embeddings = model(sequence_tokens=tokens).embeddings.cpu()

    for i, seq_id in enumerate(batch_ids):
        emb = embeddings[i]
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        torch.save(emb, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

# Main loop
print("Processing sequences in batches...")
batch_ids, batch_seqs = [], []
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or len(seq) >= 30000:
        continue

    batch_ids.append(seq_id)
    batch_seqs.append(seq)

    if len(batch_ids) == BATCH_SIZE:
        try:
            process_batch(batch_ids, batch_seqs)
        except Exception as e:
            print(f"Error processing batch {batch_ids}: {e}")
        batch_ids, batch_seqs = [], []

# Final partial batch
if batch_ids:
    try:
        process_batch(batch_ids, batch_seqs)
    except Exception as e:
        print(f"Error processing final batch {batch_ids}: {e}")

print(f"Done. Embeddings saved in: {OUTPUT_DIR}")
