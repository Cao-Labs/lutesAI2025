# This script feeds protein sequences into ESM-3 and derives sequence embeddings

import os
import torch
import esm
from pathlib import Path

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
OUTPUT_DIR = "/data/summer2020/naufal/esm3_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("Loading ESM-3 now..")
model, alphabet = esm.pretrained.esm3_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # Use .cpu() if no GPU

# Helper: FASTA reader
def fasta_reader(path):
    with open(path, "r") as file:
        identifier, sequence = None, []
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

# Stream process
print("Feeding sequences...")
for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence).")
        continue

    batch_labels, batch_strs, batch_tokens = batch_converter([(seq_id, seq)])
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33][0, 1: len(seq) + 1]
    output_path = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
    torch.save(token_representations.cpu(), output_path)

    if idx % 100 == 0:
        print(f"Processed {idx} sequences...")

print(f"Completed. Embeddings saved to: {OUTPUT_DIR}")

