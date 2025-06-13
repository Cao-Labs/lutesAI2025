import os
import torch
import numpy as np

# CONFIG
INPUT_DIR = "/data/summer2020/naufal/esm3_embeddings"
MAX_FILES = 10          # How many .pt files to show
SAVE_TO_TEXT = False    # Save output as .txt? (True/False)

def pretty_print(tensor: torch.Tensor, filename: str):
    array = tensor.numpy()
    print(f"\n=== {filename} ===")
    print(f"Shape: {array.shape}")
    
    if np.isnan(array).any():
        print("⚠️  WARNING: NaN values found in this file.")
    else:
        print("No NaNs detected.")

    # Print the first few rows of the array
    print("Preview:")
    preview_rows = min(10, array.shape[0])
    print(np.array2string(array[:preview_rows], precision=3, suppress_small=True))

    if SAVE_TO_TEXT:
        out_path = os.path.join(INPUT_DIR, filename + ".txt")
        with open(out_path, "w") as f:
            f.write(f"# {filename}\n")
            f.write(f"# Shape: {array.shape}\n")
            f.write("# NaNs present: " + str(np.isnan(array).any()) + "\n\n")
            np.savetxt(f, array, fmt="%.4f")
        print(f"(Saved to {out_path})")

# Main loop
print("Scanning directory for .pt files...")
pt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")]
if not pt_files:
    print("No .pt files found in the directory.")
else:
    for idx, fname in enumerate(pt_files[:MAX_FILES]):
        path = os.path.join(INPUT_DIR, fname)
        tensor = torch.load(path)
        pretty_print(tensor, fname)
