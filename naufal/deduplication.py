import os
import torch
import hashlib

# Path to directory with .pt files
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
BATCH_SIZE = 10000

# Track hashes we've seen
seen_hashes = {}
duplicates = 0
processed = 0

print("Starting deduplication (batch size = 10,000)...\n")

# Get all .pt filenames
all_filenames = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".pt")]
total_files = len(all_filenames)

for i in range(0, total_files, BATCH_SIZE):
    batch_filenames = all_filenames[i:i + BATCH_SIZE]

    for fname in batch_filenames:
        fpath = os.path.join(EMBEDDINGS_DIR, fname)

        try:
            tensor = torch.load(fpath, map_location="cpu")
            tensor_bytes = tensor.numpy().tobytes()
            tensor_hash = hashlib.sha1(tensor_bytes).hexdigest()

            if tensor_hash in seen_hashes:
                os.remove(fpath)
                duplicates += 1
            else:
                seen_hashes[tensor_hash] = fname

            processed += 1

        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    print(f"Batch {i // BATCH_SIZE + 1}: Processed {min(i + BATCH_SIZE, total_files)} / {total_files} files")

# Final summary
print("\nFinished deduplication.")
print(f"Total files processed: {processed}")
print(f"Duplicates removed: {duplicates}")
print(f"Unique embeddings retained: {len(seen_hashes)}")
