import os
import torch
import numpy as np

# Path to your embeddings folder
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings_new"

expected_D = 1536
min_val, max_val = float("inf"), float("-inf")
mismatched_shape_count = 0
nan_or_inf_count = 0
total_files = 0

print("Checking embedding dimensions, value ranges, and presence of NaNs/Infs...\n")

for filename in os.listdir(EMBEDDINGS_DIR):
    if not filename.endswith(".pt"):
        continue

    filepath = os.path.join(EMBEDDINGS_DIR, filename)
    total_files += 1

    try:
        tensor = torch.load(filepath)
        shape = tensor.shape

        if len(shape) != 2 or shape[1] != expected_D:
            mismatched_shape_count += 1
            continue  # skip value range and NaN/Inf check for malformed tensors

        np_tensor = tensor.numpy()

        if np.isnan(np_tensor).any() or np.isinf(np_tensor).any():
            nan_or_inf_count += 1
            continue  # skip value range updates for bad tensors

        # Update global min/max
        min_val = min(min_val, np.min(np_tensor))
        max_val = max(max_val, np.max(np_tensor))

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# Summary report
print(f"Total files checked: {total_files}")
print(f"Files with shape != [L, {expected_D}]: {mismatched_shape_count}")
print(f"Files containing NaN or Inf: {nan_or_inf_count}")

if mismatched_shape_count + nan_or_inf_count < total_files:
    print(f"\nGlobal value range (for valid tensors):")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
else:
    print("\nNo valid embeddings found to compute value range.")

