# processdata.py

# This script reads protein info one line at a time from large files,
# and extracts sequence, structure (PDB), and matched ID info only.
# It is optimized to use as little memory as possible.

import os

# File paths
uniprot_file = "/data/shared/databases/UniProt2025/training_data_processedUniprot_DB.txt"
mapping_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
pdb_dir = "/data/shared/databases/alphaFold/"
output_dir = "/data/summer2020/naufal/"

# Output files
fasta_out = os.path.join(output_dir, "protein_sequences.fasta")
structure_out = os.path.join(output_dir, "protein_structures.txt")
id_out = os.path.join(output_dir, "matched_ids.txt")

# Read mapping: create dict {UniProtKB-ID → Internal ID}
id_map = {}
with open(mapping_file, "r") as mapfile:
    for line in mapfile:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            uniprot_id, internal_id = parts
            id_map[internal_id] = uniprot_id

# Initialize counters
matched = 0
reported = False  # tracks whether we've reported the first 5000

# Open output files in write mode
with open(fasta_out, "w") as fasta_f, \
     open(structure_out, "w") as struct_f, \
     open(id_out, "w") as id_f, \
     open(uniprot_file, "r") as uni_f:

    # Process the UniProt file line by line
    for line in uni_f:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            continue  # skip any malformed lines

        internal_id, tax_id, go_terms, sequence = parts

        # Only process if we have a UniProtKB-ID mapping
        if internal_id in id_map:
            uniprot_id = id_map[internal_id]

            # Write sequence in FASTA format (for ESM-3)
            fasta_f.write(f">{internal_id}\n{sequence}\n")

            # Write matched internal ID
            id_f.write(f"{internal_id}\n")

            # Try to find and write the corresponding structure
            pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
            pdb_path = os.path.join(pdb_dir, pdb_filename)

            if os.path.isfile(pdb_path):
                with open(pdb_path, "r") as pdb_f:
                    for pdb_line in pdb_f:
                        if pdb_line.startswith("ATOM"):
                            struct_f.write(pdb_line)

            # Update and report match count
            matched += 1
            if matched == 5000 and not reported:
                print("✅ Matched 5000 proteins...")
                reported = True
            elif matched > 0 and matched % 100000 == 0:
                print(f"✅ Matched {matched} proteins...")

print(f"✅ Finished processing. Total matched proteins: {matched}")
