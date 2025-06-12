# This script adds GO terms to each protein in matched_ids.txt
# It removes BP, MF, CC suffixes from the GO terms
# and processes everything line by line to save memory

input_matched_ids = "/data/summer2020/naufal/matched_ids.txt"
uniprot_file = "/data/shared/databases/UniProt2025/training_data_processedUniprot_DB.txt"
output_file = "/data/summer2020/naufal/matched_ids_with_go.txt"

# Step 1: Read all matched protein IDs into a set
matched_ids_set = set()
with open(input_matched_ids, "r") as f:
    for line in f:
        matched_ids_set.add(line.strip())

# Step 2: Create a dictionary for GO terms (cleaned)
go_lookup = {}

with open(uniprot_file, "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue  # skip malformed lines
        uniprot_id, _, go_raw, _ = parts

        if uniprot_id in matched_ids_set:
            if go_raw.strip() == "":
                go_lookup[uniprot_id] = "NA"
            else:
                # Keep only the GO IDs, drop the ,BP or ,MF or ,CC
                go_terms = [item.split(',')[0] for item in go_raw.split(';') if ',' in item]
                go_lookup[uniprot_id] = ';'.join(go_terms) if go_terms else "NA"

# Step 3: Write output
count = 0
with open(input_matched_ids, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        protein_id = line.strip()
        go_terms = go_lookup.get(protein_id, "NA")
        outfile.write(f"{protein_id}\t{go_terms}\n")

        count += 1
        if count == 5000 or count % 100000 == 0:
            print(f"Processed {count} proteins...")

print("Done. Cleaned GO terms written to matched_ids_with_go.txt")

