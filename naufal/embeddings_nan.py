import os
import torch
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the ESM-3 inference model (start with GPU)
print("Loading ESM-3 inference model on GPU...")
model = ESM3InferenceClient("esm3_t33_650M_UR50D").to("cuda")  # or "cpu" if no GPU

# FASTA reader
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

# Constants
MAX_LENGTH = 30000

# Process sequences
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        continue

    if len(seq) >= MAX_LENGTH:
        print(f"Skipping {seq_id} (length {len(seq)} ≥ {MAX_LENGTH})")
        continue

    try:
        protein = ESMProtein(sequence=seq)

        try:
            # Run on GPU first
            output = model([protein], GenerationConfig())

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM for {seq_id}. Retrying on CPU...")
                torch.cuda.empty_cache()
                model = model.to("cpu")
                output = model([protein], GenerationConfig())
            else:
                raise

        # Extract residue-level sequence embeddings
        embedding = output[0].representations["residue"]  # shape: [L, D]

        # Clean: replace inf/nan, round to 3 decimals
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
        torch.save(embedding, out_path)

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Embeddings saved to: {OUTPUT_DIR}")

