import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from paths import FASTA_FILE, OUTPUT_DIR

print("Loading ESM-3 model...")
# Instantiate the 1.4B parameter model
model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda"))
model.eval()

def fasta_reader(path):
    with open(path, "r") as f:
        identifier, seq = None,
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if identifier:
                    yield identifier, "".join(seq)
                identifier = line[1:]
                seq =
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
            # Step 1: Use the correct API method and sampling configuration
            output = model.forward_and_sample(
                protein_tensor, 
                SamplingConfig(return_per_residue_embeddings=True)
            )
            
            # Step 2: Extract the tensor, move to CPU, and slice off BOS  and EOS [-1] tokens
            embeddings = output.per_residue_embedding[1:-1, :].cpu()
            
            # Save the clean tensor
            torch.save(embeddings, os.path.join(OUTPUT_DIR, f"{seq_id}.pt"))
            print(f"Successfully processed {seq_id} - Shape: {embeddings.shape}")
            
    except Exception as e:
        print(f"Error processing {seq_id}: {e}")
