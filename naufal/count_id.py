# === File path ===
mapping_file = "/data/summer2020/naufal/matched_ids_with_go.txt"

# === Set to hold unique GO terms ===
unique_go_terms = set()

# === Read the file and collect terms ===
with open(mapping_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        _, terms = parts
        for term in terms.split(";"):
            term = term.strip()
            if term:
                unique_go_terms.add(term)

# === Print result ===
print(f"Total unique GO terms: {len(unique_go_terms)}")
