import os
import torch
from huggingface_hub import login
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Authenticate with Hugging Face
login()  # Ensure your HF token is set via CLI or environment variable

# Load ESM-3 inference model
print("Loading ESM-3 model...")
model = ESM3InferenceClient.from_pretrained("esm3-open").to("cuda")  # or "cpu" if needed

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        protein = ESMProtein(sequence=seq)

        # Generate sequence embeddings (not structure)
        results = model.generate(
            [protein],
            GenerationConfig(track="sequence", num_steps=8, temperature=0.7)
        )
        protein_out = results[0]

        if hasattr(protein_out, "error"):
            print(f"Failed to generate {seq_id}: {protein_out.error}")
            continue

        if not hasattr(protein_out, "representations") or "sequence" not in protein_out.representations:
            print(f"No sequence representation found for {seq_id}, skipping.")
            continue

        embedding = torch.tensor(protein_out.representations["sequence"])  # shape: [L, D]
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000  # round to 3 decimals

        torch.save(embedding, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Sequence embeddings saved in: {OUTPUT_DIR}")




