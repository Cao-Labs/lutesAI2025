# 🧬 Protein Function Description with ESM-3 + BLIP-2

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
python esm3_batch_embedder.py
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
| 5    | Prepare GO/UniProt reference descriptions               | (manual/external)           | ✅     |
| 6    | Measure similarity (cosine distance via transformer)    | `dif_description.py`        | ✅     |
| 7    | Evaluate and output similarity score                    | `eval_similarity.py`        | ✅     |

## 📁 Project Files

- `esm3_batch_embedder.py` → Extracts embeddings from protein sequences
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
## ⚠️ License and Usage Notice

This project uses Meta AI’s ESM-3 model for protein embedding generation **strictly for academic and non-commercial research purposes**.

Please be aware that the ESM-3 model and weights may have licensing restrictions that limit commercial use and redistribution. Users should consult the official Meta ESM repository for detailed license terms: https://github.com/facebookresearch/esm

If you plan to use this work beyond academic research, please ensure compliance with all licensing and usage terms.

