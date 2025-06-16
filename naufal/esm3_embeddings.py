import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Load the model client
print("Loading ESM-3 model...")
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device="cuda")

# Input and output paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
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

# Process sequences
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or len(seq) >= 30000:
        print(f"Skipping {seq_id} (invalid or too long)")
        continue

    try:
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)

        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )

        emb = output.per_residue_embedding
        emb = torch.tensor(emb)

        # Clean NaNs and Infs
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        emb = torch.round(emb * 1000) / 1000  # Round to 3 decimal places

        torch.save(emb, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 50 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. All valid sequence embeddings saved in: {OUTPUT_DIR}")



