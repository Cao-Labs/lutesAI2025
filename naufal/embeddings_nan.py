import os
import torch
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Authenticate with Hugging Face
login()  # Assumes you have a token set via CLI or env variable

# Load the model (GPU if available)
print("Loading ESM-3 model with sequence track...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_sequence_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fasta reader
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

# Parameters
MAX_LENGTH = 30000
NUM_STEPS = 8
TEMPERATURE = 0.7

# Process
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
            # Use track="sequence" to generate sequence embeddings
            protein = model.generate(
                protein,
                GenerationConfig(
                    track="sequence",
                    num_steps=NUM_STEPS,
                    temperature=TEMPERATURE
                )
            )

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM for {seq_id}. Retrying on CPU...")
                torch.cuda.empty_cache()
                model = model.to("cpu")
                protein = model.generate(
                    protein,
                    GenerationConfig(track="sequence", num_steps=NUM_STEPS, temperature=TEMPERATURE)
                )
            else:
                raise

        # Extract final hidden states from sequence generation
        embedding = torch.tensor(protein.representations["sequence"])  # shape: [L, D]

        # Clean: replace inf/nan, round
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000  # 3 decimal places

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
        torch.save(embedding, out_path)

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Embeddings saved to: {OUTPUT_DIR}")

