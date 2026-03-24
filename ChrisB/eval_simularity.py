import os
import glob
import pandas as pd
import torch
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

print("[+] Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def structure_to_function(text):
    rules = {
        "domain": "functional domain binding activity",
        "motif": "repeating interaction motif",
        "interaction": "protein interaction",
        "binding": "binding function",
        "fold": "protein folding",
        "signal": "cell signaling"
    }
    for k, v in rules.items():
        if k in text:
            text += " " + v
    return text


def parse_go_obo(path):
    go = {}
    with open(path, 'r') as f:
        term = {}
        for line in f:
            line = line.strip()
            if line == "[Term]":
                term = {}
            elif line.startswith("id: GO:"):
                term["id"] = line.split("id: ")[1]
            elif line.startswith("name: "):
                term["name"] = line.split("name: ")[1]
            elif line.startswith('def: "'):
                term["def"] = line.split('def: "')[1].split('"')[0]
            elif line == "" and "id" in term and "def" in term:
                go[term["id"]] = {
                    "name": term.get("name", ""),
                    "definition": term["def"]
                }
    return go


def get_scores(text, go_terms):
    text = structure_to_function(clean_text(text))
    emb = model.encode(text, convert_to_tensor=True)

    scores = []
    for go_id, entry in go_terms.items():
        ref = clean_text(entry["definition"])
        ref_emb = model.encode(ref, convert_to_tensor=True)
        score = util.cos_sim(emb, ref_emb).item()
        scores.append(score)

    return scores


if __name__ == "__main__":
    go_path = "/DATA/shared/database/UniProt2025/GO_June_1_2025.obo"
    go_terms = parse_go_obo(go_path)

    files = sorted(glob.glob("*_description.txt"))

    all_scores = []

    for f in files:
        with open(f) as fp:
            text = fp.read()

        scores = get_scores(text, go_terms)

        # 🔥 stats
        stats = {
            "file": f,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "max": np.max(scores),
            "min": np.min(scores)
        }

        print(f"\n{f}")
        print(stats)

        all_scores.append(stats)

    df = pd.DataFrame(all_scores)
    df.to_csv("evaluation_summary.csv", index=False)

    print("\nSaved → evaluation_summary.csv")
