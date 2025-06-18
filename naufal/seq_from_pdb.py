import os

# Inputs
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
PDB_DIR = "/data/shared/databases/alphaFold/"
query_internal_id = "001R_FRG3G"  # <<< Change this to your desired protein ID

# Step 1: Find corresponding accession from mapping file
accession = None
with open(ID_MAPPING_FILE, "r") as f:
    for line in f:
        acc, internal = line.strip().split("\t")
        if internal == query_internal_id:
            accession = acc
            break

if accession is None:
    print(f"Error: Could not find accession for ID {query_internal_id}")
    exit(1)

# Step 2: Locate PDB file
pdb_filename = f"AF-{accession}-F1-model_v4.pdb"
pdb_path = os.path.join(PDB_DIR, pdb_filename)
if not os.path.exists(pdb_path):
    print(f"Error: PDB file {pdb_filename} not found.")
    exit(1)

# Step 3: Extract sequence from ATOM records
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z',
    'XAA': 'X', 'UNK': 'X'
}

sequence = []
seen_residues = set()

with open(pdb_path, "r") as f:
    for line in f:
        if line.startswith("ATOM"):
            resname = line[17:20].strip()
            resnum = line[22:26].strip()
            chain = line[21].strip()
            resid = (chain, resnum)

            if resid not in seen_residues:
                seen_residues.add(resid)
                one_letter = three_to_one.get(resname, 'X')
                sequence.append(one_letter)

seq_str = ''.join(sequence)
print(f"Sequence for {query_internal_id} (from {pdb_filename}):")
print(seq_str)
print(f"\nLength: {len(seq_str)} residues")
