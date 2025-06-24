import os
import torch
import math

# === Config ===
EMBEDDINGS_DIR = "/data/archives/naufal/final_embeddings"

# === Stats ===
min_L = float("inf")
max_L = float("-inf")
min_D = float("inf")
max_D = float("-inf")

global_min_value = float("inf")
global_max_value = float("-inf")

has_nan = False
has_inf = False

# === Scan files ===
print(f"[INFO] Scanning directory: {EMBEDDINGS_DIR}")
files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".pt")]
print(f"[INFO] Found {len(files)} .pt files")

for idx, fname in enumerate(files, 1):
    path = os.path.join(EMBEDDINGS_DIR, fname)
    try:
        tensor = torch.load(path)
        if not isinstance(tensor, torch.Tensor):
            print(f"[WARN] Skipped non-tensor file: {fname}")
            continue

        L, D = tensor.shape
        min_L = min(min_L, L)
        max_L = max(max_L, L)
        min_D = min(min_D, D)
        max_D = max(max_D, D)

        min_val = tensor.min().item()
        max_val = tensor.max().item()

        global_min_value = min(global_min_value, min_val)
        global_max_value = max(global_max_value, max_val)

        if torch.isnan(tensor).any():
            has_nan = True
        if torch.isinf(tensor).any():
            has_inf = True

        if idx % 10000 == 0:
            print(f"[✓] Checked {idx} files...")

    except Exception as e:
        print(f"[ERROR] Failed to load {fname}: {e}")

# === Report ===
print("\n====== Summary Report ======")
print(f"Total files checked: {len(files)}")
print(f"Min L (length):       {min_L}")
print(f"Max L (length):       {max_L}")
print(f"Min D (embedding dim): {min_D}")
print(f"Max D (embedding dim): {max_D}")
print(f"Min value in data:     {global_min_value}")
print(f"Max value in data:     {global_max_value}")
print(f"Contains NaNs?         {'Yes' if has_nan else 'No'}")
print(f"Contains Infs?         {'Yes' if has_inf else 'No'}")
print("================================")

