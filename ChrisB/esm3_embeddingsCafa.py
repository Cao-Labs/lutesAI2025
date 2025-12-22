# esm3_batch_embedder_cafa.py
import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from paths import FASTA_FILE, OUTPUT_DIR

print("Loading ESM-3 model...")
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval().to(torch.float32)

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

for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or len(seq) >= 30000:
        continue
    try:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        with torch.no_grad():
            output = model(sequence_tokens=protein_tensor.sequence[None]).embeddings.detach().cpu()
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            output = torch.round(output * 1000) / 1000
            torch.save(output[0], os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))
    except Exception as e:
        print(f"Error processing {seq_id}: {e}")
