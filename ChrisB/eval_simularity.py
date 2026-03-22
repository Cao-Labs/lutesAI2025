import os
import glob
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 🔹 FORCE CPU
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

print("[+] Loading similarity model on CPU...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


# ============================================================
# 🔹 CLEAN TEXT FUNCTION (KEY PATCH)
# ============================================================
# Removes noise that hurts similarity scores
def clean_text(text):
    text = text.lower()

    # Remove unwanted phrases (from previous outputs)
    text = text.replace("generated protein function description:", "")

    # Remove emojis / non-ascii characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# 🔹 Parse GO file
# ============================================================
def parse_go_obo(obo_path):
    go_terms = {}
    with open(obo_path, 'r', encoding='utf-8') as f:
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
                go_terms[term["id"]] = {
                    "name": term.get("name", ""),
                    "definition": term["def"]
                }
    return go_terms


# ============================================================
# 🔹 Similarity scoring (with cleaning)
# ============================================================
def get_top_go_matches(generated_desc, reference_go_dict, top_k=5):

    # 🔥 CLEAN BEFORE EMBEDDING (BIG IMPACT)
    generated_desc = clean_text(generated_desc)

    gen_embedding = model.encode(generated_desc, convert_to_tensor=True)

    results = []
    for go_id, entry in reference_go_dict.items():

        # Clean GO definitions too (important!)
        clean_def = clean_text(entry["definition"])

        ref_embedding = model.encode(clean_def, convert_to_tensor=True)
        score = util.cos_sim(gen_embedding, ref_embedding).item()

        results.append((go_id, entry["name"], entry["definition"], score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:top_k]


# ============================================================
# 🔹 Main
# ============================================================
if __name__ == "__main__":

    go_path = "/DATA/shared/database/UniProt2025/GO_June_1_2025.obo"

    print("\n[+] Loading GO terms...")
    all_go_terms = parse_go_obo(go_path)
    print(f"Loaded {len(all_go_terms)} GO terms.")

    caption_files = sorted(glob.glob("*_description.txt"))

    if not caption_files:
        print("No caption files found")
        exit()

    all_results = []

    for fpath in caption_files:
        with open(fpath, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        base = os.path.basename(fpath)
        print(f"\nEvaluating {base} ...")

        top_matches = get_top_go_matches(caption, all_go_terms)

        for go_id, name, definition, score in top_matches:
            all_results.append({
                "file": base,
                "generated_caption": caption,
                "go_id": go_id,
                "go_name": name,
                "go_definition": definition,
                "similarity_score": score
            })

    out_path = "top_similarity_results.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)

    print(f"\nSaved → {os.path.abspath(out_path)}")
