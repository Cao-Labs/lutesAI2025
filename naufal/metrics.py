import os
import math
from collections import defaultdict, deque

# === Step 1: Load GO Graph from OBO file ===
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

# === Step 2: Compute distance from root nodes ===
def compute_depths(graph, roots={"GO:0003674", "GO:0008150", "GO:0005575"}):
    depths = {}
    queue = deque((r, 0) for r in roots)
    while queue:
        node, d = queue.popleft()
        if node in depths:
            continue
        depths[node] = d
        for child in [c for c in graph if node in graph[c]]:
            queue.append((child, d + 1))
    return depths

# === Step 3: Propagate GO terms upward in DAG ===
def get_ancestors(term, graph):
    visited = set()
    stack = [term]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, []))
    return visited

# === Step 4: Load GO annotations from 2-column file ===
def load_annotations(path):
    annotations = {}
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or "\t" not in line:
                print(f"[WARN] Skipping malformed line {line_num}: {line}")
                continue
            try:
                pid, terms = line.split("\t", 1)
                annotations[pid] = set(t for t in terms.split(";") if t)
            except ValueError:
                print(f"[WARN] Could not parse line {line_num}: {line}")
                continue
    return annotations

# === Step 5: Evaluate semantic-aware precision/recall/F1/Smin ===
def evaluate(pred, true, graph, depths):
    total_prec, total_rec, total_f1, total_smin = 0, 0, 0, 0
    TP, FP, FN = 0, 0, 0
    count = 0

    for pid in pred:
        if pid not in true:
            continue

        pred_terms = set()
        true_terms = set()

        for go in pred[pid]:
            pred_terms |= get_ancestors(go, graph)
        for go in true[pid]:
            true_terms |= get_ancestors(go, graph)

        inter = pred_terms & true_terms
        p = len(inter) / len(pred_terms) if pred_terms else 0
        r = len(inter) / len(true_terms) if true_terms else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        fn_terms = true_terms - pred_terms
        fp_terms = pred_terms - true_terms

        fn_depth = sum(depths.get(t, 0) for t in fn_terms)
        fp_depth = sum(depths.get(t, 0) for t in fp_terms)
        smin = math.sqrt(fn_depth**2 + fp_depth**2)

        total_prec += p
        total_rec += r
        total_f1 += f1
        total_smin += smin

        TP += len(inter)
        FP += len(fp_terms)
        FN += len(fn_terms)
        count += 1

    avg_prec = total_prec / count if count else 0
    avg_rec = total_rec / count if count else 0
    avg_f1 = total_f1 / count if count else 0
    avg_smin = total_smin / count if count else 0

    print("\n[SEMANTIC METRICS]")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print(f"Smin:      {avg_smin:.4f}")
    print(f"TP: {TP}  FP: {FP}  FN: {FN}")

# === Entry point ===
def main():
    pred_file = "test_pred.txt"  # Format: <ID>\t<GO1;GO2;GO3>
    true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
    obo_file = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"

    print("[INFO] Loading GO graph...")
    graph = load_go_graph(obo_file)
    depths = compute_depths(graph)

    print("[INFO] Loading annotations...")
    pred = load_annotations(pred_file)
    true = load_annotations(true_file)

    print("[INFO] Evaluating...")
    evaluate(pred, true, graph, depths)

if __name__ == "__main__":
    main()





