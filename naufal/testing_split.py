import os
import random
import shutil

# Paths
FASTA_FILE = "/data/summer2020/naufal/protein_sequences.fasta"
PDB_DIR = "/data/shared/databases/alphaFold/"
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"

OUT_SEQ_DIR = "/data/summer2020/naufal/testing_sequences"
OUT_PDB_DIR = "/data/summer2020/naufal/testing_pdbs"
MASTER_FASTA = os.path.join(OUT_SEQ_DIR, "testing_sequences.fasta")

os.makedirs(OUT_SEQ_DIR, exist_ok=True)
os.makedirs(OUT_PDB_DIR, exist_ok=True)

# Step 1: Parse protein_sequences.fasta line-by-line
print("Reading protein_sequences.fasta...")
sequences = {}
with open(FASTA_FILE, "r") as f:
    current_id, seq_lines = None, []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_id and seq_lines:
                sequences[current_id] = "".join(seq_lines)
            current_id = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)
    if current_id and seq_lines:
        sequences[current_id] = "".join(seq_lines)
print(f"Loaded {len(sequences)} sequences.\n")

# Step 2: Random 20% test set
random.seed(42)
all_ids = list(sequences.keys())
num_test = int(0.2 * len(all_ids))
test_ids = set(random.sample(all_ids, num_test))

# Step 3: Write master FASTA + per-sequence files immediately
print("Writing sequence files...")
written = 0
with open(MASTER_FASTA, "w") as master_file:
    for pid in test_ids:
        seq = sequences[pid]
        # Master FASTA
        master_file.write(f">{pid}\n{seq}\n")
        # Individual FASTA
        with open(os.path.join(OUT_SEQ_DIR, f"{pid}.fasta"), "w") as indiv_file:
            indiv_file.write(f">{pid}\n{seq}\n")
        written += 1
        if written == 1 or written % 10000 == 0:
            print(f"Written {written} sequence files...")

print(f"Finished writing {written} sequences.\n")

# Step 4: Stream through mapping file and copy PDBs immediately
print("Copying matching PDBs...")
copied, missing = 0, 0
with open(ID_MAPPING_FILE, "r") as f:
    for line in f:
        acc, internal_id = line.strip().split("\t")
        if internal_id not in test_ids:
            continue
        pdb_filename = f"AF-{acc}-F1-model_v4.pdb"
        src_path = os.path.join(PDB_DIR, pdb_filename)
        dst_path = os.path.join(OUT_PDB_DIR, pdb_filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
            if copied == 1 or copied % 10000 == 0:
                print(f"Copied {copied} PDB files...")
        else:
            missing += 1

# Final summary
print("\nTesting dataset creation complete.")
print(f"Sequences written: {written}")
print(f"PDBs copied: {copied}")
print(f"PDBs missing: {missing}")

