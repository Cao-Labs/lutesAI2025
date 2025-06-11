# protein_features.py
# This script extracts secondary structure and surface accessibility
# for each protein using the installed DSSP binary (not Biopython).
# It processes one protein at a time from the structure file and writes residue-wise features.

import os
import subprocess
from tempfile import NamedTemporaryFile

# === File paths ===
structure_file = "/data/summer2020/naufal/protein_structures.txt"
id_file = "/data/summer2020/naufal/matched_ids.txt"
output_file = "/data/summer2020/naufal/protein_features.txt"
dssp_exec = "/data/shared/tools/DeepQA/tools/dsspcmbi"  # Your professor's installed DSSP

# === Load internal UniProt IDs ===
with open(id_file, "r") as f:
    matched_ids = [line.strip() for line in f if line.strip()]

# === Open output file ===
with open(output_file, "w") as out_f:
    current_lines = []
    protein_index = 0

    def run_dssp(internal_id, pdb_lines):
        # Write temp PDB file
        with NamedTemporaryFile(mode='w+', suffix=".pdb", delete=False) as tmp_pdb:
            tmp_pdb.writelines(pdb_lines)
            tmp_pdb_path = tmp_pdb.name

        tmp_dssp_path = tmp_pdb_path + ".dssp"

        try:
            # Call external DSSP binary
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

            # Parse DSSP block: starts after line beginning with "  #  RESIDUE"
            start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    start = i + 1
                    break

            if start is None:
                print(f"Skipped {internal_id}: DSSP output malformed")
                return

            out_f.write(f"# {internal_id}\n")
            for line in lines[start:]:
                if len(line) < 35:
                    continue  # skip short lines
                try:
                    ss = line[16].strip()
                    ss = ss if ss in ('H', 'E') else 'C'
                    rsa = float(line[35:38].strip())
                    out_f.write(f"{ss}\t{rsa:.3f}\n")
                except Exception:
                    continue  # skip malformed lines

        finally:
            os.remove(tmp_pdb_path)
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

    # === Read structure file one line at a time ===
    with open(structure_file, "r") as sf:
        for line in sf:
            if line.startswith("ATOM") and line[6:11].strip() == "1":
                # New protein starts
                if current_lines:
                    if protein_index < len(matched_ids):
                        run_dssp(matched_ids[protein_index], current_lines)
                        protein_index += 1
                    current_lines = []

            current_lines.append(line)

        # Handle last protein
        if current_lines and protein_index < len(matched_ids):
            run_dssp(matched_ids[protein_index], current_lines)

print("Done: Features written to protein_features.txt")
