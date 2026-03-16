import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import random
import torch
import esm
from sklearn.decomposition import PCA

# -----------------------------
# CONFIG
# -----------------------------

ASPECT = "P"
WINDOW_SIZE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CFG:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    TRAIN_SEQS = os.path.join(PROJECT_ROOT, "Train", "train_sequences.fasta")
    TRAIN_TERMS = os.path.join(PROJECT_ROOT, "Train", "train_terms.tsv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_plots")

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD ESM MODEL
# -----------------------------

print("Loading ESM model...")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

model = model.to(DEVICE)
model.eval()

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------

def get_esm_embeddings(seq):

    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    embeddings = results["representations"][33][0,1:-1].cpu().numpy()

    return embeddings


# -----------------------------
# PCA REDUCTION
# -----------------------------

def reduce_dimensions(embeddings, n_components=20):

    if len(embeddings) < 5:
        return embeddings

    try:
        pca = PCA(n_components=min(n_components, len(embeddings), embeddings.shape[1]))
        return pca.fit_transform(embeddings)

    except:
        return embeddings[:, :n_components]


# -----------------------------
# HYDROPHOBICITY TRACE
# -----------------------------

def get_sliding_window_trace(seq, window=20):

    hydro_map = {
        "A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
        "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
        "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2
    }

    vals = [hydro_map.get(aa,0) for aa in seq]

    if len(vals) < window:
        return [np.mean(vals)]

    means = []

    for i in range(len(vals)-window+1):
        means.append(np.mean(vals[i:i+window]))

    return means


# -----------------------------
# ONE HOT ENCODING
# -----------------------------

def get_one_hot(seq, max_len=1000):

    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    vocab = {aa:i for i,aa in enumerate(aa_order)}

    mat = np.zeros((len(aa_order), min(len(seq), max_len)))

    for i, aa in enumerate(seq[:max_len]):
        if aa in vocab:
            mat[vocab[aa], i] = 1

    return mat, aa_order


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("Loading Data...")

    seqs = {}

    for rec in SeqIO.parse(CFG.TRAIN_SEQS, "fasta"):
        pid = rec.id.split("|")[1] if "|" in rec.id else rec.id.split()[0]
        seqs[pid] = str(rec.seq)

    train_terms = pd.read_csv(CFG.TRAIN_TERMS, sep="\t")

    aspect_df = train_terms[train_terms["aspect"] == ASPECT]

    prot_terms = aspect_df.groupby("EntryID")["term"].apply(list).to_dict()

    valid_pids = [p for p in seqs if p in prot_terms]

    selected_pids = random.sample(valid_pids, 10)

    print("Selected proteins:", selected_pids)

    for pid in selected_pids:

        print("Generating plot for", pid)

        seq = seqs[pid]
        terms = prot_terms[pid]

        one_hot, aa_labels = get_one_hot(seq)

        trace = get_sliding_window_trace(seq)

        esm_emb = get_esm_embeddings(seq)

        esm_pca = reduce_dimensions(esm_emb).T

        fig, axes = plt.subplots(
            4,1,
            figsize=(12,14),
            gridspec_kw={"height_ratios":[2,2,1,1]}
        )

        sns.heatmap(one_hot, ax=axes[0], cmap="Blues", cbar=False,
                    yticklabels=list(aa_labels))

        axes[0].set_title(f"{pid} One-Hot Encoding")

        sns.heatmap(esm_pca, ax=axes[1], cmap="viridis", cbar=False)

        axes[1].set_title("ESM Embedding PCA")

        axes[2].plot(trace, color="orange")
        axes[2].axhline(0, linestyle="--", color="gray")
        axes[2].set_title("Hydrophobicity Trace")

        axes[3].axis("off")

        import textwrap

        wrapped = textwrap.fill(", ".join(terms), width=80)

        axes[3].text(0.1,0.5,wrapped,fontsize=12)

        plt.tight_layout()

        save_path = os.path.join(CFG.OUTPUT_DIR, f"viz_{pid}.png")

        plt.savefig(save_path)

        plt.close()

        print("Saved", save_path)

    print("Done")


if __name__ == "__main__":
    main()
