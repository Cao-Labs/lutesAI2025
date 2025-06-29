import os
import subprocess
from tempfile import NamedTemporaryFile

# === File paths ===
PDB_DIR = "/data/summer2020/naufal/testing_pdbs"
OUTPUT_FILE = "/data/summer2020/naufal/testing_features.txt"
DSSP_EXEC = "/data/shared/tools/DeepQA/tools/dsspcmbi"

# === Output writer ===
with open(OUTPUT_FILE, "w") as out_f:
    pdb_files = sorted(f for f in os.listdir(PDB_DIR) if f.endswith(".pdb"))
    print(f"[✓] Found {len(pdb_files)} PDB files.")

    for pdb_file in pdb_files:
        internal_id = pdb_file[:-4]  # remove .pdb
        pdb_path = os.path.join(PDB_DIR, pdb_file)

        # Run DSSP
        tmp_dssp_path = pdb_path + ".dssp"
        try:
            result = subprocess.run(
                [DSSP_EXEC, pdb_path, tmp_dssp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0 or not os.path.exists(tmp_dssp_path):
                print(f"[!] Skipped {internal_id}: DSSP failed")
                continue

            with open(tmp_dssp_path, "r") as dssp_f:
                lines = dssp_f.readlines()

            start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    start = i + 1
                    break
            if start is None:
                print(f"[!] Skipped {internal_id}: DSSP output malformed")
                continue

            # Parse DSSP output
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

            # Write to output file
            out_f.write(f"# {internal_id}\n")
            for ss, rsa in zip(ss_list, rsa_list):
                out_f.write(f"{ss}\t{rsa:.3f}\n")

            print(f"[✓] Processed {internal_id}")

        finally:
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

print(f"[✓] All features saved to {OUTPUT_FILE}")

