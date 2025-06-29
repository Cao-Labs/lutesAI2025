import os
import subprocess
from tempfile import NamedTemporaryFile

# === Paths ===
PDB_DIR = "/data/summer2020/naufal/testing_pdbs"
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
OUTPUT_FILE = "/data/summer2020/naufal/testing_features.txt"
DSSP_EXEC = "/data/shared/tools/DeepQA/tools/dsspcmbi"

# === Step 1: Load UniProt Accession → Internal ID mapping ===
accession_to_internal = {}
with open(ID_MAPPING_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            accession, _, internal = parts
            accession_to_internal[accession] = internal

print(f"[✓] Loaded {len(accession_to_internal)} accession → internal ID mappings")

# === Step 2: Process PDB files and run DSSP ===
with open(OUTPUT_FILE, "w") as out_f:
    pdb_files = sorted(f for f in os.listdir(PDB_DIR) if f.endswith(".pdb"))
    print(f"[✓] Found {len(pdb_files)} PDB files")

    for pdb_file in pdb_files:
        # Extract UniProt accession from PDB filename
        # Format: AF-<accession>-F1-model_v4.pdb
        try:
            parts = pdb_file.split("-")
            accession = parts[1]
            if accession not in accession_to_internal:
                print(f"[!] Skipped {pdb_file}: accession {accession} not in mapping")
                continue
            internal_id = accession_to_internal[accession]
        except:
            print(f"[!] Skipped malformed PDB filename: {pdb_file}")
            continue

        pdb_path = os.path.join(PDB_DIR, pdb_file)
        tmp_dssp_path = pdb_path + ".dssp"

        try:
            result = subprocess.run(
                [DSSP_EXEC, pdb_path, tmp_dssp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0 or not os.path.exists(tmp_dssp_path):
                print(f"[!] DSSP failed for {internal_id}")
                continue

            with open(tmp_dssp_path, "r") as dssp_f:
                lines = dssp_f.readlines()

            start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    start = i + 1
                    break
            if start is None:
                print(f"[!] DSSP output malformed for {internal_id}")
                continue

            ss_list = []
            rsa_list = []
            for line in lines[start:]:
                if len(line) < 38:
                    continue
                try:
                    ss = line[16].strip()
                    ss = ss if ss in ("H", "E") else "C"
                    rsa = float(line[35:38].strip())
                    ss_list.append(ss)
                    rsa_list.append(rsa)
                except:
                    continue

            out_f.write(f"# {internal_id}\n")
            for ss, rsa in zip(ss_list, rsa_list):
                out_f.write(f"{ss}\t{rsa:.3f}\n")

            print(f"[✓] Processed {internal_id}")

        finally:
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

print(f"[✓] All DSSP features saved to {OUTPUT_FILE}")

