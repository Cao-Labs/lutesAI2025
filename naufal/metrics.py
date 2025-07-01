import os
import math
import json
from collections import defaultdict, deque

# === File paths ===
pred_file = "test_pred.txt"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
obo_file = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"

# === Step 1: Parse GO DAG ===
def load_go_graph(obo_path):
    graph = defaultdict(set)
    with open(obo_path) as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("is_a:") and current_id:
                parent = line.split("is_a: ")[1].split()[0]
                graph[current_id].add(parent)
    return graph

# === Step 2: Compute distance from GO root nodes ===
def compute_depths(go_graph, roots=None):
    if roots is None:
        roots = {"GO:0003674", "GO:0008150", "GO:0005575"}  # MF, BP, CC

    depth = {}
    queue = deque((r, 0) for r in roots)
    visited = set()

    while queue:
        node, d = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        depth[node] = d
        for child in [n for n in go_graph if node in go_graph[n]]:
            queue.append((child, d + 1))
    return depth

# === Step 3: Recursively get all ancestors of a GO term ===
def get_ancestors(term, graph):
    visited = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, []))
    return visited

# === Step 4: Load annotation files ===
def load_annotations(file_path):
    annots = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pid, term_str = parts
            terms = set(t for t in term_str.split(";") if t)
            annots[pid] = terms
    return annots

# === Step 5: Evaluation using propagation + root-distance semantics ===
def evaluate_semantic(pred_annots, true_annots, go_graph, go_depths):
    total_prec, total_rec, total_f1, total_smin = 0, 0, 0, 0
    TP, FP, FN = 0, 0, 0
    count = 0

    for pid in pred_annots:
        if pid not in true_annots:
            continue

        pred_terms = pred_annots[pid]
        true_terms = true_annots[pid]

        # Propagate
        pred_full = set()
        true_full = set()
        for t in pred_terms:
            pred_full |= get_ancestors(t, go_graph)
        for t in true_terms:
            true_full |= get_ancestors(t, go_graph)

        # Semantic similarity using common ancestors
        common = pred_full & true_full
        p = len(common) / len(pred_full) if pred_full else 0
        r = len(common) / len(true_full) if true_full else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

        # Semantic Smin using root distances as proxy for IC
        fn_terms = true_full - pred_full
        fp_terms = pred_full - true_full
        fn_ic = sum(go_depths.get(t, 0) for t in fn_terms)
        fp_ic = sum(go_depths.get(t, 0) for t in fp_terms)
        smin = math.sqrt(fn_ic ** 2 + fp_ic ** 2)

        TP += len(common)
        FP += len(fp_terms)
        FN += len(fn_terms)

        total_prec += p
        total_rec += r
        total_f1 += f1
        total_smin += smin
        count += 1

    avg_prec = total_prec / count if count else 0.0
    avg_rec = total_rec / count if count else 0.0
    avg_f1 = total_f1 / count if count else 0.0
    avg_smin = total_smin / count if count else 0.0

    print("\n[Evaluation Using GO Root Distance Semantics]")
    print(f"Proteins evaluated: {count}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print(f"Smin:      {avg_smin:.4f}")
    print(f"TP: {TP}  FP: {FP}  FN: {FN}")

# === Main execution ===
if __name__ == "__main__":
    print("[INFO] Loading GO graph...")
    go_graph = load_go_graph(obo_file)
    go_depths = compute_depths(go_graph)

    print("[INFO] Loading prediction and true labels...")
    pred_annots = load_annotations(pred_file)
    true_annots = load_annotations(true_file)

    print("[INFO] Running semantic evaluation...")
    evaluate_semantic(pred_annots, true_annots, go_graph, go_depths)




