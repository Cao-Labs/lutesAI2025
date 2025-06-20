import os
import subprocess
from tempfile import NamedTemporaryFile

# === File paths ===
structure_file = "/data/summer2020/naufal/protein_structures.txt"
id_file = "/data/summer2020/naufal/matched_ids.txt"
fasta_file = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
output_file = "/data/summer2020/naufal/features_seq_aligned_to_fasta.txt"
dssp_exec = "/data/shared/tools/DeepQA/tools/dsspcmbi"

# === Load sequences from FASTA ===
sequence_dict = {}
with open(fasta_file, "r") as f:
    current_id = None
    seq = []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_id:
                sequence_dict[current_id] = "".join(seq)
            current_id = line[1:]
            seq = []
        else:
            seq.append(line)
    if current_id:
        sequence_dict[current_id] = "".join(seq)

# === Load protein IDs ===
with open(id_file, "r") as f:
    matched_ids = [line.strip() for line in f if line.strip()]

# === Output file ===
with open(output_file, "w") as out_f:
    current_lines = []
    protein_index = 0

    def run_dssp_by_index(internal_id, pdb_lines, sequence):
        # Write temp PDB
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
                print(f"Skipped {internal_id}: DSSP failed")
                return

            with open(tmp_dssp_path, "r") as dssp_f:
                lines = dssp_f.readlines()

            start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    start = i + 1
                    break
            if start is None:
                print(f"Skipped {internal_id}: DSSP output malformed")
                return

            # Extract DSSP SS/RSA per index
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

            # Align by sequence index
            out_f.write(f"# {internal_id}\n")
            for i in range(len(sequence)):
                if i < len(ss_list):
                    out_f.write(f"{ss_list[i]}\t{rsa_list[i]:.3f}\n")
                else:
                    out_f.write("X\t-1.000\n")

        finally:
            os.remove(tmp_pdb_path)
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

    # === Process structures ===
    with open(structure_file, "r") as sf:
        for line in sf:
            if line.startswith("ATOM") and line[6:11].strip() == "1":
                if current_lines:
                    if protein_index < len(matched_ids):
                        pid = matched_ids[protein_index]
                        if pid in sequence_dict:
                            run_dssp_by_index(pid, current_lines, sequence_dict[pid])
                        protein_index += 1
                    current_lines = []
            current_lines.append(line)

        # Final protein
        if current_lines and protein_index < len(matched_ids):
            pid = matched_ids[protein_index]
            if pid in sequence_dict:
                run_dssp_by_index(pid, current_lines, sequence_dict[pid])

print("Done: Aligned DSSP features written to", output_file)


