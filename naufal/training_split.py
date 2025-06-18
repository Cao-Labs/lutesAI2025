import os

# Paths
TEST_SEQ_DIR = "/data/summer2020/naufal/testing_sequences"
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"

# Collect protein IDs by removing '.fasta' extension from each file
print("Scanning testing_sequences folder...")
testing_ids = {
    os.path.splitext(filename)[0]
    for filename in os.listdir(TEST_SEQ_DIR)
    if filename.endswith(".fasta")
}
print(f"Found {len(testing_ids)} protein IDs in testing set.\n")

# Delete matching .pt files from embeddings
deleted, missing = 0, 0
print("Deleting matching embedding files...")

for pid in testing_ids:
    pt_file_path = os.path.join(EMBEDDINGS_DIR, f"{pid}.pt")
    if os.path.exists(pt_file_path):
        os.remove(pt_file_path)
        deleted += 1
        if deleted == 1:
            print(f"Deleted 1 embedding: {pid}")
        elif deleted % 10000 == 0:
            print(f"Deleted {deleted} embeddings so far...")
    else:
        missing += 1

# Final summary
print("\nDone.")
print(f"Embeddings deleted: {deleted}")
print(f"Missing embeddings (not found): {missing}")

