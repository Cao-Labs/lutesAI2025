import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Load the model client
print("Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)

# Input and output paths
FASTA_FILE = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
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

# Batch processing
print("Processing sequences in batches of 32...")
batch_size = 32
batch = []
count = 0

def process_batch(batch):
    global count
    try:
        proteins = [ESMProtein(sequence=seq) for _, seq in batch]
        encoded = [model.encode(p) for p in proteins]
        sequence_batch = torch.stack([e.sequence for e in encoded])

        with torch.no_grad():
            outputs = model(sequence_tokens=sequence_batch).embeddings.detach().cpu()

        for i, (seq_id, _) in enumerate(batch):
            out = outputs[i]
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = torch.round(out * 1000) / 1000
            torch.save(out, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        count += len(batch)
        if count % 50 == 0 or count == len(batch):
            print(f"Processed {count} sequences...")

    except Exception as e:
        print(f"Error processing batch starting with {batch[0][0]}: {e}")

# Stream through the FASTA file
for seq_id, seq in fasta_reader(FASTA_FILE):
    if not seq or len(seq) >= 30000:
        print(f"Skipping {seq_id} (invalid or too long)")
        continue

    batch.append((seq_id, seq))
    if len(batch) == batch_size:
        process_batch(batch)
        batch = []

# Final incomplete batch
if batch:
    process_batch(batch)

print(f"Done. All valid sequence embeddings saved in: {OUTPUT_DIR}")
