# === Count unique protein IDs in test_pred.txt ===

pred_file = "/data/shared/github/lutesAI2025/naufal/test_pred.txt"

unique_ids = set()

with open(pred_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        pid = parts[0]
        unique_ids.add(pid)

print(f"[✓] Number of unique protein IDs in test_pred.txt: {len(unique_ids)}")
