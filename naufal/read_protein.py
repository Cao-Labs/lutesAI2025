import torch
import numpy as np
import os

# Path to the file
FILE = "/data/archives/naufal/final_embeddings/PF2R_MOUSE.pt"

# Load the tensor
if not os.path.exists(FILE):
    print("File not found:", FILE)
    exit()

tensor = torch.load(FILE)

# Convert to numpy
array = tensor.numpy()

# Print basic info
print(f"File: {os.path.basename(FILE)}")
print(f"Shape: {array.shape}")

# Check for NaN and Inf
has_nan = np.isnan(array).any()
has_inf = np.isinf(array).any()

if has_nan:
    print("⚠️  WARNING: This tensor contains NaN values.")
else:
    print("No NaN values found.")

if has_inf:
    print("⚠️  WARNING: This tensor contains Inf values.")
else:
    print("No Inf values found.")

# Preview first few rows
print("\nFirst 10 rows:")
print(np.array2string(array[:10], precision=3, suppress_small=True))

