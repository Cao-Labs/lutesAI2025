# This script feeds protein sequences into ESM-3 and derives sequence embeddings

import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# File paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ESM-3 model via Hugging Face
print("Loading ESM-3 model from Hugging Face...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")  # Use "cpu" if needed

# Generator to read sequences one at a time
def fasta_reader(path):
    with open(path, "r") as file:
        identifier, sequence = None, []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if identifier:
                    yield identifier, "".join(sequence)
                identifier = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if identifier:
            yield identifier, "".join(sequence)

# Process sequences
print("Generating structure embeddings...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence).")
        continue

    try:
        protein = ESMProtein(sequence=seq)
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        coords = torch.tensor(protein.coordinates)  # [L, 3, 3]
        torch.save(coords, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. All embeddings saved to {OUTPUT_DIR}")


