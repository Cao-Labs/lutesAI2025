import os

# Paths
PDB_DIR = "/data/shared/databases/alphaFold"
IDMAP_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
OUTPUT_FASTA = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
OUTPUT_DIR = os.path.dirname(OUTPUT_FASTA)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Standard amino acid mapping (3-letter to 1-letter codes)
aa3_to_aa1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Helper: extract sequence from PDB ATOM lines
def extract_sequence_from_pdb(pdb_path):
    sequence = []
    seen_residues = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = line[22:26].strip()
                res_name = line[17:20].strip()
                if res_num not in seen_residues:
                    seen_residues.add(res_num)
                    aa = aa3_to_aa1.get(res_name, "X")
                    sequence.append(aa)
    return "".join(sequence)

# Helper: find internal ID from accession by scanning mapping file
def find_internal_id_from_accession(accession):
    with open(IDMAP_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                acc, internal_id = parts
                if acc == accession:
                    return internal_id
    return None

# Stream PDB directory and write sequences
print("Processing PDB files...")
written = 0
with open(OUTPUT_FASTA, "w") as out_fasta:
    for fname in os.listdir(PDB_DIR):
        if not fname.endswith(".pdb"):
            continue

        try:
            parts = fname.split("-")
            accession = parts[1]
        except IndexError:
            continue

        internal_id = find_internal_id_from_accession(accession)
        if not internal_id:
            continue

        full_path = os.path.join(PDB_DIR, fname)
        sequence = extract_sequence_from_pdb(full_path)

        if sequence:
            out_fasta.write(f">{internal_id}\n{sequence}\n")
            written += 1
            if written == 1 or written % 10000 == 0:
                print(f"Written {written} sequences...")

print(f"\nDone. Total sequences written: {written}")
