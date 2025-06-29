import os
import subprocess
from tqdm import tqdm

# === Paths ===
PDB_DIR = "/data/summer2020/naufal/testing_pdbs"
ID_MAPPING_FILE = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
OUTPUT_FILE = "/data/summer2020/naufal/testing_features.txt"
DSSP_EXEC = "/data/shared/tools/DeepQA/tools/dsspcmbi"

# === Ensure output directory exists ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# === Output file ===
with open(OUTPUT_FILE, "w") as out_f:
    pdb_files = sorted(f for f in os.listdir(PDB_DIR) if f.endswith(".pdb"))
    print(f"[✓] Found {len(pdb_files)} PDB files")

    count = 0
    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        # Extract UniProt accession from filename: AF-<accession>-F1-model_v4.pdb
        try:
            accession = pdb_file.split("-")[1]
        except Exception:
            print(f"[!] Skipped malformed filename: {pdb_file}")
            continue

        # === Map accession → internal ID (line-by-line) ===
        internal_id = None
        with open(ID_MAPPING_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == accession:
                    internal_id = parts[-1]
                    break

        if not internal_id:
            print(f"[!] Skipped {accession}: Not found in mapping")
            continue

        # === Run DSSP ===
        pdb_path = os.path.join(PDB_DIR, pdb_file)
        tmp_dssp_path = pdb_path + ".dssp"

        try:
            subprocess.run(
                [DSSP_EXEC, pdb_path, tmp_dssp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if not os.path.exists(tmp_dssp_path):
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
                print(f"[!] DSSP malformed for {internal_id}")
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
                except Exception:
                    continue

            out_f.write(f"# {internal_id}\n")
            for ss, rsa in zip(ss_list, rsa_list):
                out_f.write(f"{ss}\t{rsa:.3f}\n")

            count += 1
            if count == 1 or count % 10000 == 0:
                print(f"[✓] Processed {count:,} proteins")

        finally:
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

print(f"[✓] All DSSP features saved to {OUTPUT_FILE}")



