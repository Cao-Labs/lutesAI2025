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

total_processed = 0
for seq_id, seq in fasta_reader(FASTA_FILE):
    if not seq or len(seq) >= 30000:
        print(f"Skipping {seq_id} (invalid or too long)")
        continue

    batch.append((seq_id, seq))

    if len(batch) == batch_size:
        try:
            proteins = [ESMProtein(sequence=s) for _, s in batch]
            encodings = model.encode(proteins)

            with torch.no_grad():
                outputs = model(
                    sequence_tokens=encodings.sequence
                ).embeddings.detach().cpu()

            for i, (seq_id, _) in enumerate(batch):
                embedding = outputs[i]
                embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
                embedding = torch.round(embedding * 1000) / 1000
                torch.save(embedding, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

            total_processed += len(batch)
            print(f"Processed {total_processed} sequences...")
        except Exception as e:
            print(f"Error in batch: {e}")
        batch = []

# Final incomplete batch
if batch:
    try:
        proteins = [ESMProtein(sequence=s) for _, s in batch]
        encodings = model.encode(proteins)

        with torch.no_grad():
            outputs = model(
                sequence_tokens=encodings.sequence
            ).embeddings.detach().cpu()

        for i, (seq_id, _) in enumerate(batch):
            embedding = outputs[i]
            embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
            embedding = torch.round(embedding * 1000) / 1000
            torch.save(embedding, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        total_processed += len(batch)
        print(f"Processed {total_processed} sequences (final batch).")
    except Exception as e:
        print(f"Error in final batch: {e}")

print(f"Done. All valid sequence embeddings saved in: {OUTPUT_DIR}")
