import os
import torch
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

# Authenticate with Hugging Face
login()  # You can run huggingface-cli login once to skip this in the future

# Load ESM-3 model using GitHub-style call
print("Loading ESM-3 model...")
model = ESM3.from_pretrained("esm3-open").to("cuda")  # or "cpu" if no GPU

# Paths
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

# Process sequences one-by-one
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        continue

    try:
        protein = ESMProtein(sequence=seq)

        # Generate only sequence embeddings
        protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))

        if hasattr(protein, "error"):
            print(f"Error in generation for {seq_id}: {protein.error}")
            continue

        if not hasattr(protein, "representations") or "sequence" not in protein.representations:
            print(f"No sequence embedding found for {seq_id}, skipping.")
            continue

        # Clean and save embedding
        embedding = torch.tensor(protein.representations["sequence"])
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000

        torch.save(embedding, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Sequence embeddings saved in: {OUTPUT_DIR}")



