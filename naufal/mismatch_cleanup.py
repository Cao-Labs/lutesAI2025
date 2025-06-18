import os

# Paths
SEQ_DIR = "/data/summer2020/naufal/testing_sequences"
PDB_DIR = "/data/summer2020/naufal/testing_pdbs"
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"

# Step 1: Build a set of all .fasta filenames (without extension)
print("Indexing testing_sequences...")
all_seq_ids = {
    os.path.splitext(f)[0]
    for f in os.listdir(SEQ_DIR)
    if f.endswith(".fasta")
}
print(f"Found {len(all_seq_ids)} sequence files.\n")

# Step 2: Track which internal IDs have valid PDBs
print("Reading mapping file and checking PDBs...")
valid_ids = set()

with open(ID_MAPPING_FILE, "r") as f:
    for line in f:
        acc, internal_id = line.strip().split("\t")
        if internal_id in all_seq_ids:
            pdb_name = f"AF-{acc}-F1-model_v4.pdb"
            pdb_path = os.path.join(PDB_DIR, pdb_name)
            if os.path.exists(pdb_path):
                valid_ids.add(internal_id)

print(f"Found {len(valid_ids)} sequences with valid PDBs.\n")

# Step 3: Delete any .fasta file not in valid_ids
print("Deleting orphan sequence files...")
deleted, kept = 0, 0
for seq_id in all_seq_ids:
    if seq_id not in valid_ids:
        fasta_path = os.path.join(SEQ_DIR, f"{seq_id}.fasta")
        if os.path.exists(fasta_path):
            os.remove(fasta_path)
            deleted += 1
            if deleted == 1 or deleted % 1000 == 0:
                print(f"Deleted {deleted} orphan sequences...")
    else:
        kept += 1

# Summary
print("\nCleanup complete.")
print(f"Deleted: {deleted} orphan sequences (no matching PDB)")
print(f"Kept: {kept} sequences with valid PDBs")


