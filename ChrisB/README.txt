# notes:
conda create ChrisB python=3.8
pip install matplotlib pillow transformers sentence-transformers esm

pip install salesforce-lavis



python generate_protein_image.py --sequence "MASTSEQ..." --out protein_image.png

# confirm where you load the esm3 embeddings ..... 


my code to genereate embedding images, you need to download the data and store it in ../data/ -> https://www.kaggle.com/datasets/seddiktrk/cafa6-protein-embeddings-esm2 :
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import random
import torch
from transformers import AutoTokenizer, EsmModel
from sklearn.decomposition import PCA

# Configuration
ASPECT = 'P'
WINDOW_SIZE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # Use small model for speed/demo purposes. User can change to t33.

class CFG:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    TRAIN_SEQS = os.path.join(PROJECT_ROOT, 'Train', 'train_sequences.fasta')
    TRAIN_TERMS = os.path.join(PROJECT_ROOT, 'Train', 'train_terms.tsv')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis_plots')

if not os.path.exists(CFG.OUTPUT_DIR):
    os.makedirs(CFG.OUTPUT_DIR)

def get_esm2_embeddings(seq, model, tokenizer):
    """
    Returns per-residue embeddings [L, D]
    """
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False, padding=False, truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # [1, L, D] -> [L, D]
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    return embeddings

def reduce_dimensions(embeddings, n_components=20):
    if len(embeddings) < 5: return embeddings # Too short for PCA
    try:
        pca = PCA(n_components=min(n_components, len(embeddings), embeddings.shape[1]))
        reduced = pca.fit_transform(embeddings)
        return reduced
    except:
        return embeddings[:, :n_components] # Fallback

def get_sliding_window_trace(seq, window=20):
    """
    Returns the trace of Hydrophobicity means along the sequence.
    Length will be Len(seq) - window + 1.
    """
    if not seq: return []
    hydro_map = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
                 'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
                 'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    vals = [hydro_map.get(aa, 0) for aa in seq]
    
    if len(vals) < window:
        return [np.mean(vals)] # Too short
        
    means = []
    # Stride 1 for detailed visualization
    for i in range(len(vals) - window + 1):
        chunk = vals[i : i+window]
        means.append(np.mean(chunk))
    return means

def get_one_hot(seq, max_len=1000):
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    vocab = {aa: i for i, aa in enumerate(aa_order)}
    
    mat = np.zeros((len(aa_order), min(len(seq), max_len)))
    for i, aa in enumerate(seq[:max_len]):
        if aa in vocab:
            mat[vocab[aa], i] = 1
    return mat, aa_order

def main():
    print("Loading Data...")
    seqs = {}
    for rec in SeqIO.parse(CFG.TRAIN_SEQS, "fasta"):
        pid = rec.id.split('|')[1] if '|' in rec.id else rec.id.split()[0]
        seqs[pid] = str(rec.seq)

    train_terms = pd.read_csv(CFG.TRAIN_TERMS, sep='\t')
    aspect_df = train_terms[train_terms['aspect'] == ASPECT]
    prot_terms = aspect_df.groupby('EntryID')['term'].apply(list).to_dict()
    
    # Load ESM Model
    print(f"Loading ESM-2 Model ({ESM_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE)
    model.eval()

    # Filter for proteins that have both sequence and labels
    valid_pids = [p for p in seqs if p in prot_terms]
    
    # Pick 10 Random
    selected_pids = random.sample(valid_pids, 10)
    print(f"Selected 10 Proteins: {selected_pids}")
    
    for pid in selected_pids:
        print(f"Generating plot for {pid}...")
        seq = seqs[pid]
        terms = prot_terms[pid]
        
        # 1. One Hot
        one_hot, aa_labels = get_one_hot(seq)
        
        # 2. Sliding Window Trace
        trace = get_sliding_window_trace(seq, window=WINDOW_SIZE)
        
        # 3. ESM embeddings
        esm_emb = get_esm2_embeddings(seq, model, tokenizer)
        esm_pca = reduce_dimensions(esm_emb, n_components=20).T # Transpose for Heatmap [Features, Length]
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 2, 1, 1]})
        
        # Plot 1: One-Hot Heatmap
        sns.heatmap(one_hot, ax=axes[0], cmap="Blues", cbar=False, yticklabels=list(aa_labels))
        axes[0].set_title(f"Protein: {pid} (Length: {len(seq)}) - One-Hot Encoding")
        axes[0].set_ylabel("Amino Acid")
        axes[0].set_xticks([]) # Hide x-axis labels for alignment
        
        # Plot 2: ESM-2 Embedding (PCA)
        sns.heatmap(esm_pca, ax=axes[1], cmap="viridis", cbar=False)
        axes[1].set_title(f"ESM-2 Embeddings (PCA reduced to 20 dims)")
        axes[1].set_ylabel("PCA Component")
        axes[1].set_xticks([])
        
        # Plot 3: Sliding Window Hydrophobicity
        x_vals = range(len(trace))
        axes[2].plot(x_vals, trace, color='orange', linewidth=2)
        axes[2].axhline(y=0, color='gray', linestyle='--')
        axes[2].set_title(f"Sliding Window Hydrophobicity (Window={WINDOW_SIZE})")
        axes[2].set_ylabel("Hydrophobicity")
        axes[2].set_xlim(0, len(seq))
        axes[2].set_xlabel("Residue Position")
        
        # Plot 4: Text Info
        axes[3].axis('off')
        import textwrap
        wrapped_terms = textwrap.fill(", ".join(terms), width=80)
        text_content = f"True GO Terms ({ASPECT}):\n\n{wrapped_terms}"
        axes[3].text(0.1, 0.5, text_content, fontsize=12, va='center', wrap=True)
        
        plt.tight_layout()
        save_path = os.path.join(CFG.OUTPUT_DIR, f"viz_{pid}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    print(f"\nDone! Plots saved to {CFG.OUTPUT_DIR}")

if __name__ == "__main__":
    main()




## ⚠️ License and Usage Notice

This project uses Meta AI’s ESM-3 model for protein embedding generation **strictly for academic and non-commercial research purposes**.

Please be aware that the ESM-3 model and weights may have licensing restrictions that limit commercial use and redistribution. Users should consult the official Meta ESM repository for detailed license terms: https://github.com/facebookresearch/esm

If you plan to use this work beyond academic research, please ensure compliance with all licensing and usage terms.

# 🧬 Protein Function Description with ESM-3 + BLIP-2

Intro
Every cell in your body is packed with proteins — complex molecules that control everything from metabolism to immunity. But for all their importance, we still don’t know what most proteins actually do. That’s because figuring out a protein’s function usually requires expensive lab experiments or time-consuming structural analysis.

What if we could skip that? What if we could look at a protein — not under a microscope, but as an image — and ask an AI to tell us what it might do?

That’s exactly what we’re building.
We start with the raw protein sequence and feed it into a powerful AI from Meta called ESM-3, which turns the sequence into a high-dimensional vector — a kind of fingerprint of the protein. From there, we reshape this data into a 2D image — a visual representation of the protein’s hidden patterns.

Then comes the twist: instead of using a biology tool, we hand this image to BLIP‑2, a vision-language model originally trained on real-world photographs. It looks at the protein image and tries to describe what it sees — in plain English.

We’re still refining the output, but if successful, this approach could let us predict protein function without touching a microscope — just using sequence data and AI.
It’s a bridge between biology and computer vision that opens the door to faster, scalable, and more interpretable protein discovery.

<!--
This project uses Meta AI’s ESM-3 to generate embeddings from protein sequences, visualizes them as 2D images, and applies Salesforce’s BLIP-2 (via LAVIS) to produce natural-language function descriptions. These captions are then compared to GO or UniProt reference descriptions using SentenceTransformer cosine similarity.
-->

## 📦 Setup Instructions

```bash
# Clone this repo and LAVIS
git clone https://github.com/yourusername/protein-function-blip2.git
cd protein-function-blip2

git clone https://github.com/salesforce/LAVIS
cd LAVIS
conda env create -f environment.yml
conda activate lavis

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib pillow transformers sentence-transformers esm
```

## 🚀 How to Run


1. Run `esm3_batch_embedder.py` to extract embeddings from the sequences:

```bash
python esm3_embeddings.py
```

This will save `.pt` embedding files to a directory like `esm3_embeddings`.

2. Run `generate_protein_image.py` to convert a sequence to an image:

```bash
python generate_protein_image.py --sequence "MASTSEQ..." --out protein_image.png
```

3. Run `run_blip2.py` to generate a protein function caption from the image:

```bash
python run_blip2.py
```

4. Provide a reference description from GO or UniProt and compare them using `eval_similarity.py`:

```bash
python eval_similarity.py
```

Edit `eval_similarity.py` to include your descriptions or modify it for batch processing.

## 🔁 Pipeline Overview

| Step | Description                                             | Script/File                 | Status |
|------|---------------------------------------------------------|-----------------------------|--------|
| 1    | Generate ESM-3 embeddings                               | `esm3_embeddings.py`        | ✅     |
| 2    | Convert embeddings to 2D matrix                         | `generate_protein_image.py` | ✅     |
| 3    | Normalize and save as image                             | `generate_protein_image.py` | ✅     |
| 4    | Generate image caption with BLIP-2                      | `run_blip2.py`              | ✅     |
| 5    | Prepare GO/UniProt reference descriptions               | `eval_similarity.py`        | ✅     |
| 6    | Measure similarity (cosine distance via transformer)    | `dif_description.py`        | ✅     |
| 7    | Evaluate and output similarity score                    | `eval_similarity.py`        | ✅     |

## 📁 Project Files

- `esm3_embeddings.py` → Extracts embeddings from protein sequences
- `generate_protein_image.py` → Converts embeddings to normalized 2D matrices as `.png`
- `run_blip2.py` → Runs BLIP-2 to generate natural language captions from protein images
- `dif_description.py` → Contains `similarity_score()` using SentenceTransformer
- `eval_similarity.py` → Compares a generated caption and a reference description

## 🧠 Notes

<!--
- ESM-3 used is the official GitHub version from Meta (facebookresearch/esm)
- Embeddings are per-residue and converted to fixed-size square matrices
- Captions are generated from visualized embeddings (no structure or text input)
- Reference descriptions must be provided manually or queried from databases
-->

## 👤 Author

Created by **Christopher Barker**   
Pacific Lutheran University  
GitHub: [@ChristopherDSBarker](https://github.com/ChristopherDSBarker)  
Email: Christopher.Barker@plu.edu





