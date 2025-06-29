import os
import torch
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# === Paths ===
SOURCE_DIR = "/data/summer2020/naufal/testing_normalized"
DEST_DIR = "/data/summer2020/naufal/testing_pca"
os.makedirs(DEST_DIR, exist_ok=True)

# === Config ===
N_TOKENS_ORIG = 1913
N_FEATURES_ORIG = 1541
N_TOKENS_PCA = 512
N_FEATURES_PCA = 512
MAX_FILES_FOR_PCA = 1000  # use for PCA fitting only

# === Step 1: Collect tensors for PCA fitting ===
print("[✓] Collecting tensors for PCA fitting...")
all_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".pt"))
fit_files = all_files[:MAX_FILES_FOR_PCA]

token_matrix = []
for fname in tqdm(fit_files, desc="Loading PCA fit set"):
    path = os.path.join(SOURCE_DIR, fname)
    tensor = torch.load(path)
    if tensor.shape != (N_TOKENS_ORIG, N_FEATURES_ORIG):
        continue
    token_matrix.append(tensor.numpy())

token_matrix = np.stack(token_matrix)  # (N, 1913, 1541)
print(f"[✓] Loaded {token_matrix.shape[0]} samples for PCA.")

# === Step 2: Fit PCA on token axis ===
print("[✓] Fitting PCA on token dimension...")
flattened = token_matrix.transpose(0, 2, 1).reshape(-1, N_TOKENS_ORIG)  # (N×1541, 1913)
pca_tokens = PCA(n_components=N_TOKENS_PCA, svd_solver="randomized")
pca_tokens.fit(flattened)
joblib.dump(pca_tokens, os.path.join(DEST_DIR, "pca_tokens.pkl"))

# === Step 3: Transform token axis for next PCA ===
tokens_reduced = pca_tokens.transform(flattened)
tokens_reduced = tokens_reduced.reshape(-1, N_FEATURES_ORIG, N_TOKENS_PCA).transpose(0, 2, 1)  # (N, 512, 1541)

# === Step 4: Fit PCA on feature axis ===
print("[✓] Fitting PCA on feature dimension...")
flattened_feats = tokens_reduced.reshape(-1, N_FEATURES_ORIG)  # (N×512, 1541)
pca_features = PCA(n_components=N_FEATURES_PCA, svd_solver="randomized")
pca_features.fit(flattened_feats)
joblib.dump(pca_features, os.path.join(DEST_DIR, "pca_features.pkl"))

# === Step 5: Apply PCA to all files ===
print("[✓] Applying PCA to full dataset...")
count = 0

for fname in tqdm(all_files, desc="Reducing and saving"):
    in_path = os.path.join(SOURCE_DIR, fname)
    out_path = os.path.join(DEST_DIR, fname)

    try:
        tensor = torch.load(in_path)
        if tensor.shape != (N_TOKENS_ORIG, N_FEATURES_ORIG):
            continue

        x = tensor.numpy()

        # Token PCA
        token_input = x.T  # (1541, 1913)
        token_reduced = pca_tokens.transform(token_input).T  # (512, 1541)

        # Feature PCA
        feat_reduced = pca_features.transform(token_reduced)  # (512, 512)

        torch.save(torch.tensor(feat_reduced, dtype=torch.float32), out_path)
        count += 1

        if count == 1 or count % 10_000 == 0:
            print(f"[✓] Saved {count:,} embeddings")

    except Exception as e:
        print(f"[!] Skipped {fname}: {e}")
        continue

print(f"[✓] All done. Saved {count:,} PCA-reduced embeddings to {DEST_DIR}")
