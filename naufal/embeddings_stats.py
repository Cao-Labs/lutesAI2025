import os
import torch

# Paths and constants
EMBEDDINGS_DIR = "/data/summer2020/naufal/esm3_embeddings"
expected_D = 1536

min_val = float("inf")
max_val = float("-inf")
mismatched_shape_count = 0
nan_or_inf_count = 0
total_files = 0
min_L = float("inf")
max_L = float("-inf")

print("Checking embedding dimensions, value ranges, and presence of NaNs/Infs...\n")

with torch.no_grad():
    for idx, filename in enumerate(os.listdir(EMBEDDINGS_DIR)):
        if not filename.endswith(".pt"):
            continue

        filepath = os.path.join(EMBEDDINGS_DIR, filename)
        total_files += 1

        try:
            tensor = torch.load(filepath, map_location="cpu")
            if tensor.ndim != 2 or tensor.shape[1] != expected_D:
                mismatched_shape_count += 1
                continue

            L = tensor.shape[0]
            min_L = min(min_L, L)
            max_L = max(max_L, L)

            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_or_inf_count += 1
                continue

            tensor_min = tensor.min().item()
            tensor_max = tensor.max().item()

            min_val = min(min_val, tensor_min)
            max_val = max(max_val, tensor_max)

            if total_files == 1:
                print("First valid file processed.")
            elif total_files % 10000 == 0:
                print(f"Processed {total_files} files...")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Summary report
print(f"\nTotal files checked: {total_files}")
print(f"Files with shape != [L, {expected_D}]: {mismatched_shape_count}")
print(f"Files containing NaN or Inf: {nan_or_inf_count}")

valid_count = total_files - mismatched_shape_count - nan_or_inf_count
if valid_count > 0:
    print(f"\nGlobal value range (for valid tensors):")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    print(f"\nObserved L (sequence length) range:")
    print(f"  Min L: {min_L}")
    print(f"  Max L: {max_L}")
else:
    print("\nNo valid embeddings found to compute value or length range.")



