import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ESM-3 model from Hugging Face
print("Loading ESM-3 model...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")  # use "cpu" if no GPU

# Generator to read sequences one at a time
def fasta_reader(fasta_path):
    with open(fasta_path, "r") as file:
        identifier = None
        sequence = []
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

# Process each protein sequence
print("Processing sequences...")
count = 0
skipped = 0
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        skipped += 1
        continue

    try:
        # Wrap sequence as ESMProtein
        protein = ESMProtein(sequence=seq)

        # Generate sequence embeddings
        protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))

        # Extract residue-wise embeddings [L, 1280]
        embeddings = protein.representations["sequence"]

        # Save tensor
        torch.save(embeddings.cpu(), os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))
        count += 1

        if count == 5000 or count % 100000 == 0:
            print(f"Saved embeddings for {count} proteins...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")
        skipped += 1

print(f"Done. Saved embeddings for {count} proteins.")
print(f"Skipped {skipped} sequences.")
print(f"Embeddings saved in: {OUTPUT_DIR}")
