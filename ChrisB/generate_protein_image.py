import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -----------------------------
# CONFIG
# -----------------------------

EMBEDDING_DIR = "/data/shared/databases/esm_embeddings"
OUTPUT_DIR = "generated_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# LOAD PRECOMPUTED ESM EMBEDDINGS
# -----------------------------

def load_esm_embedding(pid):
    """
    Loads a precomputed ESM embedding (.pt file)
    Returns numpy array [L, D]
    """

    emb_path = os.path.join(EMBEDDING_DIR, f"{pid}.pt")

    if not os.path.exists(emb_path):
        print(f"[WARNING] Missing embedding for {pid}")
        return None

    emb = torch.load(emb_path, map_location="cpu")

    # Handle common ESM storage formats
    if isinstance(emb, dict):

        if "representations" in emb:
            rep = emb["representations"]

            if isinstance(rep, dict):
                emb = list(rep.values())[0]
            else:
                emb = rep

        elif "mean_representations" in emb:
            emb = list(emb["mean_representations"].values())[0]

        elif "embedding" in emb:
            emb = emb["embedding"]

        else:
            emb = list(emb.values())[0]

    if torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()

    if not isinstance(emb, np.ndarray):
        print(f"[ERROR] Could not extract embedding tensor for {pid}")
        return None

    print(f"Loaded embedding for {pid} with shape {emb.shape}")

    return emb


# -----------------------------
# FIND AVAILABLE EMBEDDINGS
# -----------------------------

embedding_files = [
    f.replace(".pt", "")
    for f in os.listdir(EMBEDDING_DIR)
    if f.endswith(".pt")
]

print(f"Found {len(embedding_files)} embeddings")

# visualize first 10 proteins
selected_pids = embedding_files[:10]


# -----------------------------
# MAIN LOOP
# -----------------------------

for pid in selected_pids:

    print(f"\nGenerating plot for {pid}...")

    esm_emb = load_esm_embedding(pid)

    if esm_emb is None:
        continue

    # -----------------------------
    # PCA DIMENSION REDUCTION
    # -----------------------------

    try:
        n_components = min(20, esm_emb.shape[0], esm_emb.shape[1])
        pca = PCA(n_components=n_components)

        esm_pca = pca.fit_transform(esm_emb).T

    except Exception as e:
        print(f"PCA failed for {pid}: {e}")
        continue


    # -----------------------------
    # PLOT
    # -----------------------------

    fig, ax = plt.subplots(figsize=(12,6))

    sns.heatmap(
        esm_pca,
        ax=ax,
        cmap="viridis",
        cbar=False
    )

    ax.set_title(f"ESM Embedding Representation (PCA {n_components}) — {pid}")
    ax.set_ylabel("PCA Component")
    ax.set_xlabel("Residue Position")

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{pid}.png")

    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")

print("\nDone.")
