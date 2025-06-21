# find_uniprot_id.py

import os

# === Configuration ===
IDMAP_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
TARGET_ID = "COG6_ASPCL"  # Change this to any ID you want to search

# === Check and search ===
if not os.path.exists(IDMAP_FILE):
    print(f"Error: Mapping file not found at {IDMAP_FILE}")
    exit()

found = False
with open(IDMAP_FILE, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[1] == TARGET_ID:
            print(f"Match found for {TARGET_ID}:")
            print(f"UniProt Accession: {parts[0]}")
            found = True
            break

if not found:
    print(f"No match found for {TARGET_ID} in {IDMAP_FILE}")
