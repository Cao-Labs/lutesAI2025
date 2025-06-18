import os

# Paths
TEST_SEQ_DIR = "/data/summer2020/naufal/testing_sequences"
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"

# Get all protein IDs in the testing folder (strip .fasta or .txt)
print("Scanning testing_sequences folder...")
testing_ids = {
    os.path.splitext(f)[0]
    for f in os.listdir(TEST_SEQ_DIR)
    if f.endswith(".fasta") or f.endswith(".txt")
}
print(f"Found {len(testing_ids)} IDs in testing set.\n")

# Delete corresponding embedding files
deleted, missing = 0, 0
print("Deleting matching embedding files...")

for pid in testing_ids:
    pt_file = os.path.join(EMBEDDINGS_DIR, f"{pid}.pt")
    if os.path.exists(pt_file):
        os.remove(pt_file)
        deleted += 1
        if deleted == 1:
            print(f"Deleted 1 embedding: {pid}")
        elif deleted % 10000 == 0:
            print(f"Deleted {deleted} embeddings so far...")
    else:
        missing += 1

# Summary
print("\nDone.")
print(f"Embeddings deleted: {deleted}")
print(f"Missing embeddings (not found): {missing}")
