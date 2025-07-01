import os
import math
from collections import defaultdict

# === File paths ===
pred_file = "/data/shared/github/lutesAI2025/naufal/test_pred.txt"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
obo_path = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"

# === Step 1: Parse GO DAG from OBO file ===
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

# === Step 2: Propagate GO terms upward ===
def propagate_terms(go_terms, go_graph):
    visited = set()
    stack = list(go_terms)
    while stack:
        term = stack.pop()
        if term not in visited:
            visited.add(term)
            stack.extend(go_graph.get(term, []))
    return visited

# === Step 3: Load GO graph ===
go_graph = extract_go_graph(obo_path)

# === Step 4: Load true annotations with propagation ===
true_annots = {}
with open(true_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, terms = parts
        raw_terms = [t for t in terms.split(";") if t]
        propagated_terms = propagate_terms(raw_terms, go_graph)
        true_annots[pid] = set(propagated_terms)

# === Step 5: Load predictions (no propagation needed) ===
pred_annots = {}
with open(pred_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, terms = parts
        pred_annots[pid] = set(terms.split(";")) if terms else set()

# === Step 6: Compute metrics ===
TP, FP, FN = 0, 0, 0
all_precisions = []
all_recalls = []
all_fmax = []
smin_sum = 0
valid = 0

for pid, predicted in pred_annots.items():
    if pid not in true_annots:
        continue

    actual = true_annots[pid]
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fscore = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    smin = math.sqrt(fp**2 + fn**2)

    TP += tp
    FP += fp
    FN += fn
    all_precisions.append(prec)
    all_recalls.append(rec)
    all_fmax.append(fscore)
    smin_sum += smin
    valid += 1

# === Step 7: Report ===
avg_precision = sum(all_precisions) / valid if valid else 0.0
avg_recall = sum(all_recalls) / valid if valid else 0.0
avg_fmax = sum(all_fmax) / valid if valid else 0.0
avg_smin = smin_sum / valid if valid else 0.0

print(f"\n[Evaluation Metrics — Propagated GO Terms]")
print(f"Proteins evaluated: {valid}")
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall:    {avg_recall:.4f}")
print(f"Fmax:      {avg_fmax:.4f}")
print(f"Smin:      {avg_smin:.4f}")
