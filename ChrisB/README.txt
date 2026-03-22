# 🧬 Protein Function Prediction from Sequence (ESM-3 + BLIP-2)

## 📌 Overview

This project explores a new way to understand proteins using AI.

Instead of predicting function directly from sequence, we:

1. Convert a protein sequence into embeddings using ESM-3
2. Transform embeddings into a visual similarity matrix
3. Use a vision-language model (BLIP-2) to describe the image
4. Compare the description to Gene Ontology (GO) definitions

This creates a pipeline from:

```text
Sequence → Image → Language → Biological Function
```

---

## 🔁 Pipeline

```text
Protein Sequence
      ↓
ESM-3 Embeddings
      ↓
Cosine Similarity Matrix (Image)
      ↓
BLIP-2 Description
      ↓
Text Cleaning + Structure→Function Mapping
      ↓
GO Similarity Scoring
```

---

## ⚙️ Setup

### 🔹 Environment 1 (BLIP-2)

```bash
conda create -n blip2_env python=3.10
conda activate blip2_env

pip install torch torchvision torchaudio
pip install transformers==4.31.0
pip install pillow matplotlib
pip install salesforce-lavis
```

---

### 🔹 Environment 2 (Evaluation)

```bash
conda create -n esm-env python=3.10
conda activate esm-env

pip install sentence-transformers pandas
```

---

## 🚀 How to Run

### 1️⃣ Generate protein image

```bash
python generate_protein_image.py --sequence "YOUR_SEQUENCE" --out protein.png
```

---

### 2️⃣ Generate description (BLIP-2)

```bash
conda activate blip2_env
python run_blip2.py --image protein.png
```

Output:

```text
protein_description.txt
```

---

### 3️⃣ Evaluate similarity (GO)

```bash
conda activate esm-env
export CUDA_VISIBLE_DEVICES=""
python eval_similarity.py
```

Output:

```text
top_similarity_results.csv
```

---

## 📊 Output

* `*_description.txt` → generated protein description
* `top_similarity_results.csv` → similarity scores vs GO

Scores ~0.55–0.60 indicate meaningful alignment between structure and function.

---

## 🧠 Key Idea

BLIP-2 describes **structure**, but GO describes **function**.

To bridge this gap, we apply:

* text cleaning
* rule-based structure → function mapping

This improves similarity without retraining models.

---

## 📁 Files

* `generate_protein_image.py` → creates similarity matrix image
* `run_blip2.py` → generates description from image
* `eval_similarity.py` → computes GO similarity
* `top_similarity_results.csv` → final results

---

## ⚠️ Notes

* BLIP-2 and evaluation require different environments due to dependency conflicts
* GPU is optional (CPU recommended for stability)
* GO data is loaded from shared database

---

## 👤 Author

Christopher Barker
Pacific Lutheran University
GitHub: https://github.com/ChristopherDSBarker

---

## ⚠️ License

This project uses Meta AI’s ESM-3 model for academic research only.
Refer to: https://github.com/facebookresearch/esm
