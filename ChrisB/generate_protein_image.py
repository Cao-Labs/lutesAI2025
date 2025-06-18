import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

def load_esm3_model():
    print("Loading ESM-3 model (multimodal, from Meta)...")
    model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(torch.float32)
    return model

def generate_embedding(model, sequence):
    protein = ESMProtein(sequence=sequence)
    with torch.no_grad():
        result = model.forward_and_sample(
            protein,
            SamplingConfig(return_per_residue_embeddings=True)
        )
        embedding = result.per_residue_embedding  # [L x D] tensor
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    return embedding

def to_2d_matrix(embedding_matrix):
    seq_len, embed_dim = embedding_matrix.size()
    size = int(np.ceil(np.sqrt(emb
