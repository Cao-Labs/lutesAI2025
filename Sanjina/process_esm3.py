import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Input fasta file from teammate's folder
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"

# Output folder in your folder
OUTPUT_DIR = "/data/summer2020/Sanjina/esm3_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output dir if it doesn't exist

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ESM-3 model on {device}...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to(device)

def fasta_reader(fasta_path):
    with open(fasta_path, "r") as file:
        identifier = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(sequence_lines)
                identifier = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line)
        if identifier is not None:
            yield identifier, "".join(sequence_lines)

print("Processing sequences from teammate's fasta file...")

for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} due to invalid sequence")
        continue
    try:
        protein = ESMProtein(sequence=seq)
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        coords = torch.tensor(protein.coordinates)
        output_file = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
        torch.save(coords, output_file)
        print(f"[{idx}] Processed {seq_id}, saved embedding to {output_file}")
        del protein, coords
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done! Embeddings saved in: {OUTPUT_DIR}")
