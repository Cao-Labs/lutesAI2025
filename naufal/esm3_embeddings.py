import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ESM-3 model from Hugging Face (must have token set via huggingface-cli login or env var)
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
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        continue

    try:
        # Wrap sequence as ESMProtein
        protein = ESMProtein(sequence=seq)

        # Generate sequence embeddings
        results = model.generate(
            [protein],
            GenerationConfig(track="sequence", num_steps=8, temperature=0.7)
        )
        protein_out = results[0]

        # Extract sequence embedding
        if not hasattr(protein_out, "representations") or "sequence" not in protein_out.representations:
            print(f"No sequence representation found for {seq_id}, skipping.")
            continue

        embedding = torch.tensor(protein_out.representations["sequence"])  # shape: [L, D]
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000

        torch.save(embedding, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Sequence embeddings saved in: {OUTPUT_DIR}")



