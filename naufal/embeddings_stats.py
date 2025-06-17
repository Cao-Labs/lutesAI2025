import os
import torch
import numpy as np

# Path to your embeddings folder
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"

expected_D = 1536
min_val, max_val = float("inf"), float("-inf")
mismatched_shape_count = 0
nan_or_inf_count = 0
total_files = 0

min_L = float("inf")
max_L = float("-inf")

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
            continue

        L = shape[0]
        min_L = min(min_L, L)
        max_L = max(max_L, L)

        np_tensor = tensor.numpy()

        if np.isnan(np_tensor).any() or np.isinf(np_tensor).any():
            nan_or_inf_count += 1
            continue

        min_val = min(min_val, np.min(np_tensor))
        max_val = max(max_val, np.max(np_tensor))

        if total_files == 1:
            print("First valid file processed.")
        elif total_files % 50000 == 0:
            print(f"Processed {total_files} files...")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# Summary report
print(f"\nTotal files checked: {total_files}")
print(f"Files with shape != [L, {expected_D}]: {mismatched_shape_count}")
print(f"Files containing NaN or Inf: {nan_or_inf_count}")

if mismatched_shape_count + nan_or_inf_count < total_files:
    print(f"\nGlobal value range (for valid tensors):")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    print(f"\nObserved L (sequence length) range:")
    print(f"  Min L: {min_L}")
    print(f"  Max L: {max_L}")
else:
    print("\nNo valid embeddings found to compute value or length range.")


