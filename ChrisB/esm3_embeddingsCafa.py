import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Load the model client
print("Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)  # Ensure model runs in float32 to avoid dtype mismatch

# Input and output paths
FASTA_FILE =  "/data/summer2020/ChrisB/testing_sequences.fasta"  # <-- your CAFA FASTA file path
OUTPUT_DIR = "/data/summer2020/ChrisB/testing_embeddings"
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

# Process sequences
print("Processing sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or len(seq) >= 30000:
        print(f"Skipping {seq_id} (invalid or too long)")
        continue

    try:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)

        with torch.no_grad():
            output1 = model(sequence_tokens=protein_tensor.sequence[None]).embeddings.detach().cpu().numpy()[0]
            output2 = model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding.detach().cpu().numpy()

        # Clean NaNs and Infs
        output1 = torch.tensor(output1)
        output1 = torch.nan_to_num(output1, nan=0.0, posinf=0.0, neginf=0.0)
        output1 = torch.round(output1 * 1000) / 1000

        torch.save(output1, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))

        if idx % 50 == 0:
            print(f"Processed {idx} sequences...")

    except Exception as e:
        print(f"Error processing {seq_id}: {e}")

print(f"Done. All valid sequence embeddings saved in: {OUTPUT_DIR}")
