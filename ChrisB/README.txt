# Protein Function Description with BLIP-2 (LAVIS)

This project uses ESM-3 protein embeddings converted into images and BLIP-2 (via Salesforce's LAVIS repo) to generate natural language descriptions of protein function.

## 1. Clone and Set Up LAVIS

```bash
git clone https://github.com/salesforce/LAVIS
cd LAVIS
conda env create -f environment.yml
conda activate lavis

//additional
pip install torch torchvision torchaudio
pip install transformers
pip install pillow
