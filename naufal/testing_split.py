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

# Step 1: Parse FASTA into dictionary {internal_id: sequence}
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

# Step 2: Load ID mapping {accession → internal ID}
print("Loading UniProt ID mappings...")
acc_to_internal = {}
internal_to_acc = {}
with open(ID_MAPPING_FILE, "r") as f:
    for line in f:
        acc, internal_id = line.strip().split("\t")
        acc_to_internal[acc] = internal_id
        internal_to_acc[internal_id] = acc
print(f"Loaded {len(acc_to_internal)} mappings.\n")

# Step 3: Randomly sample 20% of sequence IDs
random.seed(42)
all_ids = list(sequences.keys())
num_test = int(0.2 * len(all_ids))
test_ids = set(random.sample(all_ids, num_test))

# Step 4: Write master and per-sequence FASTA files
print("Writing testing sequence files...")
with open(MASTER_FASTA, "w") as master_file:
    written = 0
    for pid in test_ids:
        seq = sequences[pid]
        # Master FASTA
        master_file.write(f">{pid}\n{seq}\n")
        # Individual .fasta
        with open(os.path.join(OUT_SEQ_DIR, f"{pid}.fasta"), "w") as indiv_file:
            indiv_file.write(f">{pid}\n{seq}\n")
        written += 1
        if written == 1 or written % 10000 == 0:
            print(f"Written {written} testing sequence files...")

print(f"Finished writing {written} sequences.\n")

# Step 5: Copy matching PDBs
print("Copying AlphaFold PDBs...")
copied, missing = 0, 0
for pid in test_ids:
    if pid not in internal_to_acc:
        missing += 1
        continue
    acc = internal_to_acc[pid]
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

# Summary
print("\nTesting dataset creation complete.")
print(f"Total sequences selected: {len(test_ids)}")
print(f"PDBs copied: {copied}")
print(f"PDBs missing: {missing}")
