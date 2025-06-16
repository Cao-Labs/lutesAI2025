import os
import torch
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Authenticate with Hugging Face
login()  # Assumes token already configured

# Load ESM-3 model (on GPU first)
print("Loading ESM-3 model...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")

# File paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_sequence_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
MAX_LENGTH = 30000
NUM_STEPS = 8
TEMPERATURE = 0.7
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# FASTA reader
def fasta_reader(path):
    with open(path, "r") as f:
        identifier = None
        seq = []
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

# Sequence processing
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid or empty sequence)")
        continue

    if len(seq) >= MAX_LENGTH:
        print(f"Skipping {seq_id} (length {len(seq)} ≥ {MAX_LENGTH})")
        continue

    if any(residue not in VALID_AA for residue in seq):
        print(f"Skipping {seq_id} (contains invalid amino acids)")
        continue

    try:
        protein = ESMProtein(sequence=seq)

        try:
            protein = model.generate(
                protein,
                GenerationConfig(track="sequence", num_steps=NUM_STEPS, temperature=TEMPERATURE)
            )

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory on {seq_id}, switching to CPU")
                torch.cuda.empty_cache()
                model = model.to("cpu")
                protein = model.generate(
                    protein,
                    GenerationConfig(track="sequence", num_steps=NUM_STEPS, temperature=TEMPERATURE)
                )
            else:
                raise

        if hasattr(protein, "error"):
            print(f"Failed to generate {seq_id}: {protein.error}")
            continue

        embedding = torch.tensor(protein.representations["sequence"])  # shape [L, D]
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000

        out_path = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
        torch.save(embedding, out_path)

        if idx % 100 == 0:
            print(f"Processed {idx} sequences")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Embeddings saved to: {OUTPUT_DIR}")


