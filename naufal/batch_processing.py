import os
import torch
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

# Load model
print("Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)

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

# Pad tensors to the same length
def pad_to_max_length(tensor_list):
    max_len = max(t.shape[0] for t in tensor_list)
    padded = []
    for t in tensor_list:
        pad_len = max_len - t.shape[0]
        padded_tensor = F.pad(t, (0, 0, 0, pad_len), value=0.0)
        padded.append(padded_tensor)
    return torch.stack(padded)

# Process sequences in batches
print("Processing sequences in batches...")
BATCH_SIZE = 8
batch = []
ids = []

def process_batch(batch, ids):
    tensors = []
    for seq in batch:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        with torch.no_grad():
            embedding = model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding.detach().cpu()
            embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
            embedding = torch.round(embedding * 1000) / 1000
            tensors.append(embedding)
    padded = pad_to_max_length(tensors)
    for i, tid in enumerate(ids):
        torch.save(padded[i], os.path.join(OUTPUT_DIR, f"{tid}.pt"))

for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or len(seq) >= 30000:
        print(f"Skipping {seq_id} (invalid or too long)")
        continue
    batch.append(seq)
    ids.append(seq_id)
    if len(batch) == BATCH_SIZE:
        try:
            process_batch(batch, ids)
            print(f"Processed batch ending at {seq_id} (total: {idx})")
        except Exception as e:
            print(f"Error processing batch ending at {seq_id}: {e}")
        batch, ids = [], []

# Process remaining
if batch:
    try:
        process_batch(batch, ids)
        print(f"Processed final batch")
    except Exception as e:
        print(f"Error processing final batch: {e}")

print(f"Done. All valid sequence embeddings saved in: {OUTPUT_DIR}")
