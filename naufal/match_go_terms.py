# This script adds GO terms to each protein ID in matched_ids.txt
# by looking them up from the UniProt dataset
# It processes everything line by line to save memory

input_matched_ids = "/data/summer2020/naufal/matched_ids.txt"
uniprot_file = "/data/shared/databases/UniProt2025/training_data_processedUniprot_DB.txt"
output_file = "/data/summer2020/naufal/matched_ids_with_go.txt"

# Step 1: Read all matched protein IDs into a set
# This will help us quickly check if a UniProt line matches one we care about
matched_ids_set = set()
with open(input_matched_ids, "r") as f:
    for line in f:
        matched_ids_set.add(line.strip())

# Step 2: Make a dictionary to hold GO terms for the matched IDs
go_lookup = {}

# Step 3: Go through the UniProt file line by line and match GO terms
with open(uniprot_file, "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue  # skip broken lines
        uniprot_id, _, go_terms, _ = parts
        if uniprot_id in matched_ids_set:
            go_lookup[uniprot_id] = go_terms

# Step 4: Go through matched_ids.txt again and write the GO terms to the new file
count = 0
with open(input_matched_ids, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        protein_id = line.strip()
        go_terms = go_lookup.get(protein_id, "NA")  # "NA" if not found
        outfile.write(f"{protein_id}\t{go_terms}\n")

        count += 1
        if count == 5000 or count % 100000 == 0:
            print(f"Processed {count} proteins...")

print("Done. Wrote GO terms to matched_ids_with_go.txt")
