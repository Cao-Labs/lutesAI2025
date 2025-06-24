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

1. Add your protein sequences to a FASTA file called `your_sequences.fasta` in this format:

```
>seq1
MASTSEQ...
>seq2
VPLAASEQ...
```

2. Run `esm3_batch_embedder.py` to extract embeddings from the sequences:

```bash
python esm3_batch_embedder.py
```

This will save `.pt` embedding files to a directory like `esm3_embeddings`.

3. Run `generate_protein_image.py` to convert a sequence to an image:

```bash
python generate_protein_image.py --sequence "MASTSEQ..." --out protein_image.png
```

4. Run `run_blip2.py` to generate a protein function caption from the image:

```bash
python run_blip2.py
```

5. Provide a reference description from GO or UniProt and compare them using `eval_similarity.py`:

```bash
python eval_similarity.py
```

Edit `eval_similarity.py` to include your descriptions or modify it for batch processing.

## 🔁 Pipeline Overview

| Step | Description                                             | Script/File                 | Status |
|------|---------------------------------------------------------|-----------------------------|--------|
| 1    | Input raw sequence (FASTA)                              | `your_sequences.fasta`      | ✅     |
| 2    | Generate ESM-3 embeddings                               | `esm3_batch_embedder.py`    | ✅     |
| 3    | Convert embeddings to 2D matrix                         | `generate_protein_image.py` | ✅     |
| 4    | Normalize and save as image                             | `generate_protein_image.py` | ✅     |
| 5    | Generate image caption with BLIP-2                      | `run_blip2.py`              | ✅     |
| 6    | Prepare GO/UniProt reference descriptions               | (manual/external)           | ✅     |
| 7    | Measure similarity (cosine distance via transformer)    | `dif_description.py`        | ✅     |
| 8    | Evaluate and output similarity score                    | `eval_similarity.py`        | ✅     |

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
