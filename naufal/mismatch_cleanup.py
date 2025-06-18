import os

# Paths
SEQ_DIR = "/data/summer2020/naufal/testing_sequences"
PDB_DIR = "/data/summer2020/naufal/testing_pdbs"
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"

# Step 1: Build internal_id → accession map
print("Reading ID mapping file...")
id_to_acc = {}
with open(ID_MAPPING_FILE, "r") as f:
    for line in f:
        acc, internal_id = line.strip().split("\t")
        id_to_acc[internal_id] = acc
print(f"Loaded {len(id_to_acc)} mappings.\n")

# Step 2: Check and delete orphan .fasta files
print("Scanning testing_sequences for unmatched sequences...")
deleted, kept = 0, 0

for fname in os.listdir(SEQ_DIR):
    if not fname.endswith(".fasta"):
        continue

    internal_id = os.path.splitext(fname)[0]
    acc = id_to_acc.get(internal_id)

    if not acc:
        os.remove(os.path.join(SEQ_DIR, fname))
        deleted += 1
        if deleted == 1 or deleted % 1000 == 0:
            print(f"Deleted {deleted} orphan sequences (no mapping found)...")
        continue

    pdb_name = f"AF-{acc}-F1-model_v4.pdb"
    if not os.path.exists(os.path.join(PDB_DIR, pdb_name)):
        os.remove(os.path.join(SEQ_DIR, fname))
        deleted += 1
        if deleted == 1 or deleted % 1000 == 0:
            print(f"Deleted {deleted} orphan sequences (no PDB found)...")
    else:
        kept += 1

# Summary
print("\nCleanup complete.")
print(f"Deleted: {deleted} orphan sequences (no matching PDB)")
print(f"Kept: {kept} sequences with valid PDBs")

