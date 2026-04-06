import os
import glob
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer

# ============================================================
#  GPU ENABLED (DIFF FROM CPU VERSION)
# ============================================================
# CPU VERSION:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# device = torch.device("cpu")

# GPU PATCH:
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[+] Loading similarity model on {device.upper()}...")

# CPU VERSION: device='cpu'
# GPU PATCH:
model = SentenceTransformer('all-mpnet-base-v2', device=device)


# ============================================================
#  CLEAN TEXT FUNCTION (UNCHANGED)
# ============================================================
def clean_text(text):
    text = text.lower()
    text = text.replace("generated protein function description:", "")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
#  STRUCTURE → FUNCTION MAPPING (UNCHANGED)
# ============================================================
def structure_to_function(text):
    rules = {
        "domain": "functional domain involved in binding or signaling",
        "motif": "repeating structural motif involved in interaction",
        "interaction": "protein protein interaction activity",
        "binding": "binding activity",
        "fold": "protein folding process",
        "disordered": "intrinsically disordered regulatory region",
        "region": "functional region",
        "signal": "cell signaling or molecular signaling process"
    }

    additions = []

    for key, val in rules.items():
        if key in text:
            additions.append(val)
            additions.append(val)

    return text + " " + " ".join(additions)


# ============================================================
# Parse GO file (UNCHANGED)
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
#  Similarity scoring (PATCHED FOR GPU MEMORY SAFETY)
# ============================================================
def get_top_go_matches(generated_desc, reference_go_dict, go_embeddings, top_k=5):

    generated_desc = clean_text(generated_desc)
    generated_desc = structure_to_function(generated_desc)

    # stays on GPU
    gen_embedding = model.encode(generated_desc, convert_to_tensor=True)
    gen_embedding = torch.nn.functional.normalize(gen_embedding, p=2, dim=0)

    results = []

    for go_id, entry in reference_go_dict.items():

        # CPU VERSION: ref_embedding = go_embeddings[go_id]
        # GPU PATCH: move ONE embedding at a time to GPU
        ref_embedding = go_embeddings[go_id].to(gen_embedding.device)

        ref_embedding = torch.nn.functional.normalize(ref_embedding, p=2, dim=0)

        score = torch.dot(gen_embedding, ref_embedding).item()

        results.append((go_id, entry["name"], entry["definition"], score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:top_k]


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":

    go_path = "/DATA/shared/database/UniProt2025/GO_June_1_2025.obo"

    print("\n[+] Loading GO terms...")
    all_go_terms = parse_go_obo(go_path)
    print(f"Loaded {len(all_go_terms)} GO terms.")

    # ========================================================
    #  PATCH: STORE EMBEDDINGS ON CPU (CRITICAL FIX)
    # ========================================================
    # CPU VERSION: everything on CPU anyway
    # BAD GPU VERSION: everything stored on GPU → OOM
    # FIX: compute with model, but STORE on CPU

    print("[+] Precomputing GO embeddings (stored on CPU)...")

    go_embeddings = {
        go_id: model.encode(
            clean_text(entry["definition"]),
            convert_to_tensor=True,
            device="cpu"   # 🔥 KEY PATCH (prevents OOM)
        )
        for go_id, entry in all_go_terms.items()
    }

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

        top_matches = get_top_go_matches(
            caption,
            all_go_terms,
            go_embeddings
        )

        for go_id, name, definition, score in top_matches:
            all_results.append({
                "file": base,
                "generated_caption": caption,
                "go_id": go_id,
                "go_name": name,
                "go_definition": definition,
                "similarity_score": score
            })

    out_path = "top_similarity_results_gpu.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)

    print(f"\nSaved → {os.path.abspath(out_path)}")
