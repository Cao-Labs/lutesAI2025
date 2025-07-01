import os
import math
import json

# === File paths ===
pred_file = "/data/shared/github/lutesAI2025/naufal/test_pred.txt"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
vocab_file = "/data/shared/github/lutesAI2025/naufal/go_vocab.json"

# === Load predictions first ===
pred_annots = {}
with open(pred_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, terms = parts
        pred_annots[pid] = set(terms.split(";")) if terms else set()

# === Load only true annotations for predicted proteins ===
true_annots = {}
with open(true_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid, terms = parts
        if pid in pred_annots:
            true_annots[pid] = set(terms.split(";"))

# === Load GO vocabulary ===
with open(vocab_file, "r") as vf:
    go_vocab = json.load(vf)
    total_vocab = len(go_vocab)

# === Initialize metrics ===
TP_total = 0
FP_total = 0
FN_total = 0
all_precisions = []
all_recalls = []
all_fmax = []
smin_sum = 0
valid = 0

# === Compute per-protein metrics ===
for pid, predicted in pred_annots.items():
    if pid not in true_annots:
        continue

    actual = true_annots[pid]
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    TP_total += tp
    FP_total += fp
    FN_total += fn

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.


