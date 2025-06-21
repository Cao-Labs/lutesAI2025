import os

# === Configuration ===
INPUT_DIR = "/data/shared/databases/alphaFold"
OUTPUT_DIR = "/data/summer2020/naufal/single_chain_pdbs"
TARGET_CHAIN = "A"
os.makedirs(OUTPUT_DIR, exist_ok=True)

processed = 0
written = 0
skipped = 0

print("Starting PDB single-chain filtering...")

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".pdb"):
        continue

    path = os.path.join(INPUT_DIR, fname)
    with open(path, "r") as f:
        lines = f.readlines()

    chains_found = set()
    for line in lines:
        if line.startswith("ATOM"):
            chains_found.add(line[21])

    if len(chains_found) <= 1 and TARGET_CHAIN in chains_found:
        # Already single-chain (and it's the right chain)
        skipped += 1
        continue

    filtered_lines = [
        line for line in lines if line.startswith("ATOM") and line[21] == TARGET_CHAIN
    ]

    if filtered_lines:
        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, "w") as out_f:
            out_f.writelines(filtered_lines)
        written += 1

    processed += 1
    if processed % 500 == 0:
        print(f"Processed: {processed} files — Written: {written}, Skipped: {skipped}")

print(f"\nDone! Total processed: {processed}, written: {written}, skipped: {skipped}")
print(f"Filtered PDBs saved to: {OUTPUT_DIR}")
