import os
import math
from collections import defaultdict

# === File paths ===
pred_file = "/data/summer2020/naufal/test_pred.txt"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"

# === Load true annotations ===
true_annots = {}
with open(true_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, terms = parts
        true_annots[pid] = set(terms.split(";"))

# === Load predictions ===
pred_annots = {}
with open(pred_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, terms = parts
        pred_annots[pid] = set(terms.split(";")) if terms else set()

# === Compute metrics ===
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

    # For Smin (simplified: use fp and fn as proxies for information content)
    smin = math.sqrt(fp**2 + fn**2)

    all_precisions.append(prec)
    all_recalls.append(rec)
    all_fmax.append(fscore)
    smin_sum += smin
    valid += 1

# === Report ===
avg_precision = sum(all_precisions) / valid if valid else 0.0
avg_recall = sum(all_recalls) / valid if valid else 0.0
avg_fmax = sum(all_fmax) / valid if valid else 0.0
avg_smin = smin_sum / valid if valid else 0.0

print(f"\n[Evaluation Metrics]")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall:    {avg_recall:.4f}")
print(f"Fmax:      {avg_fmax:.4f}")
print(f"Smin:      {avg_smin:.4f}")
