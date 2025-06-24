import os

# === Config ===
EMBEDDING_DIR = "/data/archives/naufal/final_embeddings"
MATCHED_IDS_FILE = "/data/summer2020/naufal/matched_ids_with_go.txt"

# === Step 1: Collect all protein IDs from the embedding file names ===
embedding_ids = {
    fname[:-3] for fname in os.listdir(EMBEDDING_DIR) if fname.endswith(".pt")
}
print(f"[INFO] Found {len(embedding_ids):,} embedding files (.pt)")

# === Step 2: Collect all IDs from the matched_ids_with_go.txt file ===
matched_ids = set()
with open(MATCHED_IDS_FILE, "r") as f:
    for line in f:
        if line.strip():
            pid = line.split("\t")[0].strip()
            matched_ids.add(pid)
print(f"[INFO] Found {len(matched_ids):,} matched GO ID entries")

# === Step 3: Compare ===
ids_in_both = embedding_ids & matched_ids
ids_in_embeddings_only = embedding_ids - matched_ids
ids_in_go_only = matched_ids - embedding_ids

# === Step 4: Report ===
print(f"\n[✓] Embeddings with GO terms: {len(ids_in_both):,}")
print(f"[✘] Embeddings without GO terms: {len(ids_in_embeddings_only):,}")
print(f"[✘] GO entries without embeddings: {len(ids_in_go_only):,}")

# Optional: Save lists
# with open("unmatched_embeddings.txt", "w") as out:
#     out.writelines(f"{pid}\n" for pid in sorted(ids_in_embeddings_only))

# with open("unmatched_go_ids.txt", "w") as out:
#     out.writelines(f"{pid}\n" for pid in sorted(ids_in_go_only))
