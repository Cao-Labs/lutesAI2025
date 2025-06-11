import os
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# --- File paths ---

# Input: FASTA file from  teammate's (Naufal's) folder on the server
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"

# Output: Folder in your directory for saving embeddings
OUTPUT_DIR = "/data/summer2020/Sanjina/esm3_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create folder if not present

# --- Load model ---

# Use GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ESM-3 model on {device}...")

# Load pretrained ESM-3 model (requires huggingface-cli login)
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to(device)

# --- Helper functions ---

def count_sequences(fasta_path):
    """
    Count number of sequences in the FASTA file to track progress.
    """
    count = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count

def fasta_reader(fasta_path):
    """
    Generator that yields one sequence at a time from the FASTA file.
    Keeps memory usage low by avoiding bulk loading.
    """
    with open(fasta_path, "r") as file:
        identifier = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(sequence_lines)
                identifier = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line)
        if identifier is not None:
            yield identifier, "".join(sequence_lines)

# --- Processing loop ---

print("Counting total sequences in the FASTA file...")
total_sequences = count_sequences(FASTA_FILE)
print(f"Total sequences found: {total_sequences}")
print("Starting ESM-3 embedding generation...")

for idx, (seq_id, seq) in enumerate(fasta_reader(FASTA_FILE), start=1):
    if not seq or set(seq) == {"."}:
        print(f"Skipping {seq_id} (invalid sequence)")
        continue
    try:
        # Prepare the input
        protein = ESMProtein(sequence=seq)

        # Generate structural prediction using ESM-3
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))

        # Convert to tensor and save
        coords = torch.tensor(protein.coordinates)
        output_file = os.path.join(OUTPUT_DIR, f"{seq_id}.pt")
        torch.save(coords, output_file)

        # Progress updates
        percent = (idx / total_sequences) * 100
        print(f"[{idx}/{total_sequences} | {percent:.2f}%] Processed {seq_id}")

        # Milestone update every 50,000 sequences
        if idx % 50000 == 0:
            print(f"🚀 Milestone reached: {idx} sequences processed.")

        # Free memory
        del protein, coords
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️ Error processing {seq_id}: {e}")

print(f"✅ Done! All embeddings saved to: {OUTPUT_DIR}")
