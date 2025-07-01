import os
import math
import json
from collections import defaultdict

# === File paths ===
pred_file = "/data/shared/github/lutesAI2025/naufal/test_pred.txt"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
obo_file = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"

# === Step 1: Parse GO DAG from OBO file ===
def extract_go_graph(obo_path):
    go_graph = defaultdict(set)
    parent_map = {}
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
                parent_map[current_id] = parent_map.get(current_id, []) + [parent]
    return go_graph, parent_map

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

# === Step 3: Find common ancestors using DFS ===
def find_common_ancestors(go1, go2, graph):
    ancestors1 = propagate_terms({go1}, graph)
    ancestors2 = propagate_terms({go2}, graph)
    return ancestors1.intersection(ancestors2)

# === Step 4: Load annotations ===
def load_annotations(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            pid, terms = parts
            data[pid] = set(t for t in terms.split(";") if t)
    return data

# === Load everything ===
print("[INFO] Loading GO DAG...")
go_graph, _ = extract_go_graph(obo_file)

print("[INFO] Loading predictions and true labels...")
pred_annots = load_annotations(pred_file)
true_annots = load_annotations(true_file)

# === Metrics calculation ===
TP, FP, FN = 0, 0, 0
precisions, recalls, f1s = [], [], []
smin_total = 0
valid = 0

print("[INFO] Calculating semantic-aware metrics...")
for pid, predicted in pred_annots.items():
    if pid not in true_annots:
        continue

    actual = true_annots[pid]
    predicted = propagate_terms(predicted, go_graph)
    actual = propagate_terms(actual, go_graph)

    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    smin = math.sqrt(fp**2 + fn**2)

    TP += tp
    FP += fp
    FN += fn

    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    smin_total += smin
    valid += 1

# === Final Metrics ===
avg_precision = sum(precisions) / valid if valid else 0.0
avg_recall = sum(recalls) / valid if valid else 0.0
avg_f1 = sum(f1s) / valid if valid else 0.0
avg_smin = smin_total / valid if valid else 0.0

# === Output ===
print("\n[Evaluation Metrics (Semantic-aware)]")
print(f"Total proteins evaluated: {valid}")
print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall:    {avg_recall:.4f}")
print(f"F1-score:  {avg_f1:.4f}")
print(f"Smin:      {avg_smin:.4f}")


