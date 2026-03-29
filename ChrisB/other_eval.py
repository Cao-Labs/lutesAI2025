import os
import glob
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer

# ============================================================
# FORCE CPU
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

print("[+] Loading similarity model on CPU...")
model = SentenceTransformer('all-mpnet-base-v2', device='cpu')


# ============================================================
# CLEAN TEXT FUNCTION
# ============================================================
def clean_text(text):
    text = text.lower()
    text = text.replace("generated protein function description:", "")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# STRUCTURE → FUNCTION MAPPING
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
# PARSE GO + DAG
# ============================================================
def parse_go_obo(obo_path):
    go_terms = {}
    go_dag = {}

    with open(obo_path, 'r', encoding='utf-8') as f:
        term = {"is_a": []}

        for line in f:
            line = line.strip()

            if line == "[Term]":
                term = {"is_a": []}

            elif line.startswith("id: GO:"):
                term["id"] = line.split("id: ")[1]

            elif line.startswith("name: "):
                term["name"] = line.split("name: ")[1]

            elif line.startswith('def: "'):
                term["def"] = line.split('def: "')[1].split('"')[0]

            elif line.startswith("is_a: "):
                parent_id = line.split("is_a: ")[1].split(" !")[0]
                term["is_a"].append(parent_id)

            elif line == "":
                if "id" in term:
                    go_terms[term["id"]] = {
                        "name": term.get("name", ""),
                        "definition": term.get("def", "")
                    }
                    go_dag[term["id"]] = term.get("is_a", [])

    return go_terms, go_dag


# ============================================================
# SIMILARITY SCORING (UNCHANGED CORE)
# ============================================================
def get_top_go_matches(generated_desc, reference_go_dict, go_embeddings, top_k=5):
    generated_desc = clean_text(generated_desc)
    generated_desc = structure_to_function(generated_desc)

    gen_embedding = model.encode(generated_desc, convert_to_tensor=True)
    gen_embedding = torch.nn.functional.normalize(gen_embedding, p=2, dim=0)

    results = []

    for go_id, entry in reference_go_dict.items():
        ref_embedding = go_embeddings[go_id]
        ref_embedding = torch.nn.functional.normalize(ref_embedding, p=2, dim=0)

        score = torch.dot(gen_embedding, ref_embedding).item()

        results.append((go_id, entry["name"], entry["definition"], score))

    results.sort(key=lambda x: x[3], reverse=True)
    return results[:top_k]


# ============================================================
# LOAD UNIPROT GROUND TRUTH
# ============================================================
def load_uniprot_ground_truth(filepath="uniprot_ground_truth.tsv"):
    ground_truth = {}

    if not os.path.exists(filepath):
        print(f"[!] Ground truth file not found: {filepath}")
        return ground_truth

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pid, go_id = parts[0], parts[1]

                if pid not in ground_truth:
                    ground_truth[pid] = set()

                ground_truth[pid].add(go_id)

    return ground_truth


# ============================================================
# PROPAGATE GO TERMS (DAG)
# ============================================================
def propagate_terms(terms, go_dag):
    propagated = set(terms)
    queue = list(terms)

    while queue:
        current = queue.pop(0)
        for parent in go_dag.get(current, []):
            if parent not in propagated:
                propagated.add(parent)
                queue.append(parent)

    return propagated


# ============================================================
# COMPUTE FMAX
# ============================================================
def compute_fmax(predictions_dict, ground_truth, go_dag):
    if not ground_truth:
        return 0.0

    thresholds = set()
    for preds in predictions_dict.values():
        for _, score in preds:
            thresholds.add(score)

    thresholds = sorted(thresholds, reverse=True)
    fmax = 0.0

    for t in thresholds:
        total_pr = 0.0
        total_rc = 0.0
        count = 0

        for pid, true_terms in ground_truth.items():
            if pid not in predictions_dict:
                continue

            pred_terms = [
                go_id for go_id, score in predictions_dict[pid] if score >= t
            ]

            pred_prop = propagate_terms(pred_terms, go_dag)
            true_prop = propagate_terms(true_terms, go_dag)

            if not pred_prop and not true_prop:
                continue

            tp = len(pred_prop & true_prop)
            fp = len(pred_prop - true_prop)
            fn = len(true_prop - pred_prop)

            pr = tp / (tp + fp) if (tp + fp) > 0 else 0
            rc = tp / (tp + fn) if (tp + fn) > 0 else 0

            total_pr += pr
            total_rc += rc
            count += 1

        if count > 0:
            avg_pr = total_pr / count
            avg_rc = total_rc / count

            if (avg_pr + avg_rc) > 0:
                f = 2 * avg_pr * avg_rc / (avg_pr + avg_rc)
                fmax = max(fmax, f)

    return fmax


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    go_path = "/DATA/shared/database/UniProt2025/GO_June_1_2025.obo"

    print("\n[+] Loading GO terms...")
    all_go_terms, go_dag = parse_go_obo(go_path)

    print(f"Loaded {len(all_go_terms)} GO terms.")

    print("[+] Precomputing GO embeddings...")
    go_embeddings = {
        go_id: model.encode(clean_text(entry["definition"]), convert_to_tensor=True)
        for go_id, entry in all_go_terms.items()
    }

    caption_files = sorted(glob.glob("*_description.txt"))

    if not caption_files:
        print("No caption files found")
        exit()

    all_results = []
    predictions_dict = {}
    total_sim_score = 0.0

    for fpath in caption_files:
        with open(fpath, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        base = os.path.basename(fpath)
        pid = base.replace("_description.txt", "")

        print(f"Evaluating {base}...")

        top_matches = get_top_go_matches(
            caption,
            all_go_terms,
            go_embeddings,
            top_k=5
        )

        predictions_dict[pid] = []

        if top_matches:
            total_sim_score += top_matches[0][3]

        for go_id, name, definition, score in top_matches:
            predictions_dict[pid].append((go_id, score))

            all_results.append({
                "file": base,
                "generated_caption": caption,
                "go_id": go_id,
                "go_name": name,
                "go_definition": definition,
                "similarity_score": score
            })

    pd.DataFrame(all_results).to_csv("top_similarity_results.csv", index=False)

    print("\n[+] Loading UniProt ground truth...")
    ground_truth = load_uniprot_ground_truth("uniprot_ground_truth.tsv")

    fmax_val = compute_fmax(predictions_dict, ground_truth, go_dag)
    avg_sim = total_sim_score / len(caption_files)

    print("\n=== Evaluation Results ===")
    print(f"Sentence Similarity: {avg_sim:.3f}")
    print(f"Fmax: {fmax_val:.3f}")
    print("=========================")
