import os
import subprocess
from tempfile import NamedTemporaryFile

# === Paths ===
pdb_dir = "/data/summer2020/naufal/testing_pdbs"
mapping_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
output_file = "/data/summer2020/naufal/testing_features.txt"
dssp_exec = "/data/shared/tools/DeepQA/tools/dsspcmbi"

# === Helper: Map UniProt accession to internal ID (line-by-line) ===
def map_accession_to_internal(uniprot_acc):
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3 and parts[0] == uniprot_acc:
                return parts[2]  # internal ID
    return None

# === DSSP Feature Extraction ===
def run_dssp_and_write(pdb_path, internal_id, out_f):
    with open(pdb_path, "r") as f:
        pdb_lines = f.readlines()

    with NamedTemporaryFile(mode='w+', suffix=".pdb", delete=False) as tmp_pdb:
        tmp_pdb.writelines(pdb_lines)
        tmp_pdb_path = tmp_pdb.name

    tmp_dssp_path = tmp_pdb_path + ".dssp"

    try:
        result = subprocess.run(
            [dssp_exec, tmp_pdb_path, tmp_dssp_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0 or not os.path.exists(tmp_dssp_path):
            print(f"[!] Skipped {internal_id}: DSSP failed")
            return False

        with open(tmp_dssp_path, "r") as dssp_f:
            lines = dssp_f.readlines()

        # Parse DSSP lines
        start = None
        for i, line in enumerate(lines):
            if line.startswith("  #  RESIDUE"):
                start = i + 1
                break
        if start is None:
            print(f"[!] Skipped {internal_id}: DSSP output malformed")
            return False

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

        # Write immediately
        out_f.write(f"# {internal_id}\n")
        for ss, rsa in zip(ss_list, rsa_list):
            out_f.write(f"{ss}\t{rsa:.3f}\n")

        return True

    finally:
        os.remove(tmp_pdb_path)
        if os.path.exists(tmp_dssp_path):
            os.remove(tmp_dssp_path)

# === Main Execution ===
processed = 0
with open(output_file, "w") as out_f:
    for fname in sorted(os.listdir(pdb_dir)):
        if not fname.endswith(".pdb"):
            continue

        accession = fname.split("-")[1]  # from AF-<ACC>-F1-model_v4.pdb
        internal_id = map_accession_to_internal(accession)

        if not internal_id:
            continue

        pdb_path = os.path.join(pdb_dir, fname)
        success = run_dssp_and_write(pdb_path, internal_id, out_f)

        if success:
            processed += 1
            if processed == 1:
                print(f"[✓] Processed first: {internal_id}")
            elif processed % 10_000 == 0:
                print(f"[✓] Processed {processed:,} proteins")

print(f"[✓] Done! Total proteins processed: {processed:,}")


