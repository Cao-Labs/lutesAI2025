import os
import torch
import numpy as np

# Path to your embeddings folder
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings_new"

# Tracking
expected_D = 1536
min_val, max_val = float("inf"), float("-inf")
mismatched_files = []

print("Checking embedding dimensions and value ranges...\n")

for filename in os.listdir(EMBEDDINGS_DIR):
    if filename.endswith(".pt"):
        filepath = os.path.join(EMBEDDINGS_DIR, filename)
        try:
            tensor = torch.load(filepath)
            shape = tensor.shape

            if len(shape) != 2 or shape[1] != expected_D:
                mismatched_files.append((filename, shape))

            # Update global min and max
            np_tensor = tensor.numpy()
            min_val = min(min_val, np.min(np_tensor))
            max_val = max(max_val, np.max(np_tensor))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Output summary
if mismatched_files:
    print("Files with unexpected embedding dimensions:")
    for f, s in mismatched_files:
        print(f"  {f}: shape {s}")
else:
    print("All files have consistent [L, 1536] dimensions.")

print(f"\nGlobal value range across all embeddings:")
print(f"  Min: {min_val}")
print(f"  Max: {max_val}")
