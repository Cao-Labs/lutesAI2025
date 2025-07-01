import os
import math
import json
from collections import defaultdict
from tqdm import tqdm

# === Load GO DAG from OBO file ===
def extract_go_graph(obo_path):
    go_graph = defaultdict(set)
    current_id = None
    with open(obo_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("is_a:") and current_id:
                parent = line.split("is_a: ")[1].split()[0]
                go_graph[current_id].add(parent)
    return go_graph

# === Propagate terms to ancestors ===
def get_all_ancestors(term_set, go_graph):
    visited = set()
    stack = list(term_set)
    while stack:
        term = stack.pop()
        if term not in visited:
            visited.add(term)
            stack.extend(go_graph.get(term, []))
    return visited

# === Compute semantic similarity ===
def semantic_overlap(predicted, actual, go_graph):
    pred_ancestors = get_all_ancestors(predicted, go_graph)
    actual_ancestors = get_all_ancestors(actual, go_graph)
    common = pred_ancestors & actual_ancestors
    return len(common), len(pred_ancestors), len(actual_ancestors)

# === Paths ===
obo_file = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
pred_file = "/data/shared/github/lutesAI2025/naufal/test_pred.txt"

# === Load GO graph ===
print("[INFO] Loading GO DAG...")
go_graph = extract_go_graph(obo_file)

# === Load true annotations ===
true_annots = {}
with open(true_file, "r") as f:
    for line in f:
        if "\t" not in line:
            continue
        pid, terms = line.strip().split("\t")
        true_annots[pid] = set(t for t in terms.split(";") if t)

# === Load predictions ===
pred_annots = {}
with open(pred_file, "r") as f:
    for line in f:
        if "\t" not in line:
            continue
        pid, terms = line.strip().split("\t")
        pred_annots[pid] = set(t for t in terms.split(";") if t)

# === Evaluate ===
print("[INFO] Evaluating predictions (semantic)...")
total_sim_precision = 0
total_sim_recall = 0
total_fmax = 0
total_smin = 0
valid = 0
TP = FP = FN = 0

for pid, predicted in tqdm(pred_annots.items()):
    if pid not in true_annots:
        continue
    actual = true_annots[pid]

    # Raw counts
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    TP += tp
    FP += fp
    FN += fn

    # Semantic
    common, pred_total, actual_total = semantic_overlap(predicted, actual, go_graph)
    precision = common / pred_total if pred_total else 0
    recall = common / actual_total if actual_total else 0
    fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    smin = math.sqrt((pred_total - common) ** 2 + (actual_total - common) ** 2)

    total_sim_precision += precision
    total_sim_recall += recall
    total_fmax += fscore
    total_smin += smin
    valid += 1

# === Report ===
print("\n[Semantic Evaluation Metrics]")
print(f"Valid proteins evaluated: {valid}")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision (semantic): {total_sim_precision / valid:.4f}")
print(f"Recall    (semantic): {total_sim_recall / valid:.4f}")
print(f"Fmax      (semantic): {total_fmax / valid:.4f}")
print(f"Smin      (semantic): {total_smin / valid:.4f}")

