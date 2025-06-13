import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_truncated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ESM-3 model from Hugging Face (initially on GPU)
print("Loading ESM-3 model on GPU...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")

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

# Max sequence length allowed
MAX_LENGTH = 30000

# Process each protein sequence
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        continue

    if len(seq) >= MAX_LENGTH:
        print(f"Skipping {seq_id} ({len(seq)} residues — exceeds limit of {MAX_LENGTH})")
        continue

    try:
        protein = ESMProtein(sequence=seq)

        try:
            # Attempt generation on GPU
            protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM for {seq_id}. Retrying on CPU...")
                torch.cuda.empty_cache()
                model = model.to("cpu")
                protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
            else:
                raise  # Re-raise if it's not a CUDA OOM error

        # Save rounded coordinates
        coords = torch.tensor(protein.coordinates)
        coords = torch.round(coords * 1000) / 1000  # round to 3 decimal places
        torch.save(coords, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Sequence embeddings saved in: {OUTPUT_DIR}")



