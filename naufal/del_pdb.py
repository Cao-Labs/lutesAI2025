import os

# Paths
ALPHAFOLD_DIR = "/data/shared/databases/alphaFold"
TEST_PDB_DIR = "/data/summer2020/naufal/testing_pdbs"

# Get all filenames (no extensions)
testing_files = {
    f for f in os.listdir(TEST_PDB_DIR)
    if f.endswith(".pdb")
}

print(f"Found {len(testing_files)} PDB files in testing set.\n")

# Delete from AlphaFold dir if match is found
deleted = 0
missing = 0

for pdb_file in testing_files:
    target_path = os.path.join(ALPHAFOLD_DIR, pdb_file)
    if os.path.exists(target_path):
        os.remove(target_path)
        deleted += 1
        if deleted == 1 or deleted % 10000 == 0:
            print(f"Deleted {deleted} files so far...")
    else:
        missing += 1

# Summary
print("\nDone.")
print(f"PDBs deleted from AlphaFold folder: {deleted}")
print(f"Files in testing set but not found in AlphaFold folder: {missing}")
