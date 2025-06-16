import os
import torch
from huggingface_hub import login
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Authenticate with Hugging Face
login()  # Requires "Read" permission token set via CLI or env

# Load the ESM-3 model
print("Loading ESM-3 model...")
model = ESM3InferenceClient.from_pretrained("esm3-open").to("cuda")

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

# Generation loop
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        continue

    try:
        # Create ESMProtein object
        protein = ESMProtein(sequence=seq)

        # Generate sequence representation only
        protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))

        # Check if generation succeeded and contains embeddings
        if hasattr(protein, "error"):
            print(f"Error in generation for {seq_id}: {protein.error}")
            continue
        if not hasattr(protein, "representations") or "sequence" not in protein.representations:
            print(f"No sequence embedding found for {seq_id}, skipping.")
            continue

        # Extract and clean the embedding
        embedding = torch.tensor(protein.representations["sequence"])
        embedding[torch.isinf(embedding)] = 0.0
        embedding = torch.nan_to_num(embedding, nan=0.0)
        embedding = torch.round(embedding * 1000) / 1000

        # Save to file
        out_path = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
        torch.save(embedding, out_path)

        if idx % 100 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. Sequence embeddings saved in: {OUTPUT_DIR}")



