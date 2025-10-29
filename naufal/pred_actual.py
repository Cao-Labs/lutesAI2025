# === File paths ===
pred_file = "/data/shared/github/lutesAI2025/naufal/testing_predictions.txt"
true_file = "/data/summer2020/naufal/matched_ids_with_go.txt"
output_file = "/data/shared/github/lutesAI2025/naufal/actual_predicted.txt"

# === Load actual GO annotations ===
true_annots = {}
with open(true_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            pid, terms = parts
            true_annots[pid] = terms
        else:
            true_annots[parts[0]] = ""

# === Load predicted GO annotations ===
pred_annots = {}
with open(pred_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            pid, terms = parts
            pred_annots[pid] = terms
        else:
            pred_annots[parts[0]] = ""

# === Write output file ===
with open(output_file, "w") as out:
    for pid in sorted(pred_annots.keys()):
        pred_terms = pred_annots.get(pid, "")
        true_terms = true_annots.get(pid, "")
        out.write(f"{pid}\n")
        out.write(f"Predicted: {pred_terms}\n")
        out.write(f"Actual:    {true_terms}\n")
        out.write("\n")

print(f"[✓] Output written to {output_file}")
