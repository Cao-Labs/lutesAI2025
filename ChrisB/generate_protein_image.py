# -----------------------------
# LOAD PRECOMPUTED ESM-3 EMBEDDINGS
# -----------------------------

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

    # Handle multiple possible formats
    if isinstance(emb, dict):

        if "representations" in emb:
            emb = emb["representations"]

        elif "embedding" in emb:
            emb = emb["embedding"]

        elif "mean_representations" in emb:
            emb = list(emb["mean_representations"].values())[0]

    # Convert tensor → numpy
    if torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()

    # Debug confirmation
    print(f"Loaded embedding for {pid} with shape {emb.shape}")

    return emb


# -----------------------------
# MAIN LOOP MODIFICATION
# -----------------------------

for pid in selected_pids:

    print(f"\nGenerating plot for {pid}...")

    seq = seqs[pid]
    terms = prot_terms[pid]

    # 1. One Hot Encoding
    one_hot, aa_labels = get_one_hot(seq)

    # 2. Sliding Window Hydrophobicity
    trace = get_sliding_window_trace(seq, window=WINDOW_SIZE)

    # 3. LOAD ESM-3 EMBEDDING
    esm_emb = load_esm3_embedding(pid)

    if esm_emb is None:
        print(f"Skipping {pid} (embedding not found)")
        continue

    # Reduce embedding dimensions for visualization
    esm_pca = reduce_dimensions(esm_emb, n_components=20).T


    # -----------------------------
    # PLOTTING
    # -----------------------------

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 14),
        gridspec_kw={'height_ratios': [2, 2, 1, 1]}
    )

    # Plot 1: One-Hot Heatmap
    sns.heatmap(
        one_hot,
        ax=axes[0],
        cmap="Blues",
        cbar=False,
        yticklabels=list(aa_labels)
    )

    axes[0].set_title(f"Protein: {pid} (Length: {len(seq)}) - One-Hot Encoding")
    axes[0].set_ylabel("Amino Acid")
    axes[0].set_xticks([])


    # Plot 2: ESM-3 Embedding PCA
    sns.heatmap(
        esm_pca,
        ax=axes[1],
        cmap="viridis",
        cbar=False
    )

    axes[1].set_title("ESM-3 Embedding Representation (PCA 20 components)")
    axes[1].set_ylabel("PCA Component")
    axes[1].set_xticks([])


    # Plot 3: Hydrophobicity Trace
    x_vals = range(len(trace))

    axes[2].plot(x_vals, trace, color="orange", linewidth=2)
    axes[2].axhline(y=0, color="gray", linestyle="--")

    axes[2].set_title(f"Sliding Window Hydrophobicity (Window={WINDOW_SIZE})")
    axes[2].set_ylabel("Hydrophobicity")
    axes[2].set_xlim(0, len(seq))
    axes[2].set_xlabel("Residue Position")


    # Plot 4: GO Terms
    axes[3].axis("off")

    import textwrap

    wrapped_terms = textwrap.fill(", ".join(terms), width=80)

    text_content = f"True GO Terms ({ASPECT}):\n\n{wrapped_terms}"

    axes[3].text(
        0.1,
        0.5,
        text_content,
        fontsize=12,
        va="center",
        wrap=True
    )

    plt.tight_layout()

    save_path = os.path.join(CFG.OUTPUT_DIR, f"viz_{pid}.png")

    plt.savefig(save_path)
    plt.close()

    print(f"Saved {save_path}")
