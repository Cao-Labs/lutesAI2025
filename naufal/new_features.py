import os
import subprocess
from tempfile import NamedTemporaryFile

# === File paths ===
structure_file = "/data/summer2020/naufal/protein_structures.txt"
id_file = "/data/summer2020/naufal/matched_ids.txt"
output_file = "/data/summer2020/naufal/new_features.txt"
dssp_exec = "/data/shared/tools/DeepQA/tools/dsspcmbi"

# === Load internal UniProt IDs ===
with open(id_file, "r") as f:
    matched_ids = [line.strip() for line in f if line.strip()]

# === Output file ===
with open(output_file, "w") as out_f:
    current_lines = []
    protein_index = 0

    def run_dssp(internal_id, pdb_lines):
        # Extract ordered list of residue numbers with CA atoms
        residue_order = []
        seen_res = set()
        for line in pdb_lines:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_num = line[22:26].strip()
                if res_num not in seen_res:
                    residue_order.append(res_num)
                    seen_res.add(res_num)

        if not residue_order:
            print(f"Skipped {internal_id}: No CA atoms found")
            return

        # Write temp PDB file
        with NamedTemporaryFile(mode='w+', suffix=".pdb", delete=False) as tmp_pdb:
            tmp_pdb.writelines(pdb_lines)
            tmp_pdb_path = tmp_pdb.name

        tmp_dssp_path = tmp_pdb_path + ".dssp"

        try:
            # Run DSSP
            result = subprocess.run(
                [dssp_exec, tmp_pdb_path, tmp_dssp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0 or not os.path.exists(tmp_dssp_path):
                print(f"Skipped {internal_id}: DSSP failed")
                return

            # Read DSSP output
            with open(tmp_dssp_path, "r") as dssp_f:
                lines = dssp_f.readlines()

            # Parse DSSP block
            start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    start = i + 1
                    break

            if start is None:
                print(f"Skipped {internal_id}: DSSP output malformed")
                return

            # Map DSSP values by residue number (column 6: residue number)
            ss_map = {}
            rsa_map = {}
            for line in lines[start:]:
                if len(line) < 38:
                    continue
                try:
                    res_num = line[5:10].strip()
                    ss = line[16].strip()
                    ss = ss if ss in ('H', 'E') else 'C'
                    rsa = float(line[35:38].strip())
                    ss_map[res_num] = ss
                    rsa_map[res_num] = rsa
                except Exception:
                    continue

            # Write aligned output
            out_f.write(f"# {internal_id}\n")
            for res_num in residue_order:
                ss = ss_map.get(res_num, 'X')
                rsa = rsa_map.get(res_num, -1.000)
                out_f.write(f"{ss}\t{rsa:.3f}\n")

        finally:
            os.remove(tmp_pdb_path)
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

    # === Read structure file and process each protein ===
    with open(structure_file, "r") as sf:
        for line in sf:
            if line.startswith("ATOM") and line[6:11].strip() == "1":
                if current_lines:
                    if protein_index < len(matched_ids):
                        run_dssp(matched_ids[protein_index], current_lines)
                        protein_index += 1
                    current_lines = []
            current_lines.append(line)

        # Last protein
        if current_lines and protein_index < len(matched_ids):
            run_dssp(matched_ids[protein_index], current_lines)

print("Done: Features written to protein_features.txt")
