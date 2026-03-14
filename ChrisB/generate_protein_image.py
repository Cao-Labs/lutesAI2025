# -----------------------------
# LOAD PRECOMPUTED ESM-3 EMBEDDINGS
# -----------------------------

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EMBEDDING_DIR = "/data/shared/databases/esm_embeddings"

def load_esm3_embedding(pid):
    """
    Loads a precomputed ESM-3 embedding (.pt file)
    Expected shape: [L, D]
    """

    emb_path = os.path.join(EMBEDDING_DIR, f"{pid}.pt")

    if not os.path.exists(emb_path):
        print(f"[WARNING] Missing embedding for {pid}")
        return None

    emb = torch.load(emb_path, map_location="cpu")

    if isinstance(emb, dict):

        if "representations" in emb:
            emb = emb["representations"]

        elif "embedding" in emb:
            emb = emb["embedding"]

        elif "mean_representations" in emb:
            emb = list(emb["mean_representations"].values())[0]

    if torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()

    print(f"Loaded embedding for {pid} with shape {emb.shape}")

    return emb


# -----------------------------
# FIND ALL AVAILABLE PROTEINS
# -----------------------------

embedding_files = [
    f.replace(".pt", "")
    for f in os.listdir(EMBEDDING_DIR)
    if f.endswith(".pt")
]

selected_pids = embedding_files[:10]   # process first 10 proteins


# -----------------------------
# MAIN LOOP
# -----------------------------

for pid in selected_pids:

    print(f"\nGenerating plot for {pid}...")

    esm_emb = load_esm3_embedding(pid)

    if esm_emb is None:
        continue

    # Reduce embedding dimensions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(20, esm_emb.shape[1]))
    esm_pca = pca.fit_transform(esm_emb).T


    # -----------------------------
    # PLOTTING
    # -----------------------------

    fig, axes = plt.subplots(
        1, 1,
        figsize=(12, 6)
    )

    sns.heatmap(
        esm_pca,
        ax=axes,
        cmap="viridis",
        cbar=False
    )

    axes.set_title(f"ESM-3 Embedding Representation (PCA 20 components)\nProtein: {pid}")
    axes.set_ylabel("PCA Component")
    axes.set_xlabel("Residue Position")

    plt.tight_layout()

    save_path = f"viz_{pid}.png"

    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")
