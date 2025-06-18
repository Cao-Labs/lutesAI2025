import os

# Paths
SEQ_DIR = "/data/summer2020/naufal/testing_sequences"
PDB_DIR = "/data/summer2020/naufal/testing_pdbs"

# Extract accession numbers from PDB filenames
print("Scanning testing_pdbs folder...")
pdb_accessions = {
    fname.split("-")[1]
    for fname in os.listdir(PDB_DIR)
    if fname.startswith("AF-") and fname.endswith(".pdb")
}
print(f"Found {len(pdb_accessions)} valid PDB accessions.\n")

# Build map from UniProt accession → internal ID using filenames
print("Checking sequence files for matching PDBs...")
deleted, kept = 0, 0
for fname in os.listdir(SEQ_DIR):
    if not fname.endswith(".fasta"):
        continue

    internal_id = os.path.splitext(fname)[0]
    # Find corresponding accession (reverse lookup requires mapping)
    # Extract accession from matching PDBs
    matching = False
    for acc in pdb_accessions:
        expected_pdb = f"AF-{acc}-F1-model_v4.pdb"
        if os.path.exists(os.path.join(PDB_DIR, expected_pdb)):
            expected_seq = f"{internal_id}.fasta"
            if os.path.exists(os.path.join(SEQ_DIR, expected_seq)):
                matching = True
                break

    if not matching:
        os.remove(os.path.join(SEQ_DIR, fname))
        deleted += 1
        if deleted == 1 or deleted % 1000 == 0:
            print(f"Deleted {deleted} unmatched sequences...")
    else:
        kept += 1

# Final summary
print("\nCleanup complete.")
print(f"Deleted: {deleted} orphan sequences (no PDB found)")
print(f"Kept: {kept} sequences with matching PDBs")
