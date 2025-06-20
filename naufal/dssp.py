import os
import subprocess
from tempfile import NamedTemporaryFile

# File paths
idmapping_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
sequence_file = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
pdb_dir = "/data/shared/databases/alphaFold"
dssp_exec = "/data/shared/tools/DeepQA/tools/dsspcmbi"
output_file = "/data/summer2020/naufal/final_features_no_fill.txt"

# Load UniProt internal ID → accession
print("Loading ID mapping...")
id_to_accession = {}
with open(idmapping_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            accession, internal = parts
            id_to_accession[internal] = accession

# Load sequences from FASTA
print("Loading sequences...")
seq_dict = {}
with open(sequence_file, "r") as f:
    current_id = None
    seq = []
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_id:
                seq_dict[current_id] = "".join(seq)
            current_id = line[1:]
            seq = []
        else:
            seq.append(line)
    if current_id:
        seq_dict[current_id] = "".join(seq)

# Run DSSP and write aligned output
written = 0
with open(output_file, "w") as out_f:
    for count, (internal_id, sequence) in enumerate(seq_dict.items(), start=1):
        accession = id_to_accession.get(internal_id)
        if not accession:
            continue

        pdb_file = os.path.join(pdb_dir, f"AF-{accession}-F1-model_v4.pdb")
        if not os.path.exists(pdb_file):
            print(f"Missing PDB: {accession}")
            continue

        # Write temp PDB file
        with NamedTemporaryFile(mode="w+", suffix=".pdb", delete=False) as tmp_pdb:
            with open(pdb_file, "r") as src:
                tmp_pdb.writelines(src.readlines())
            tmp_pdb_path = tmp_pdb.name

        tmp_dssp_path = tmp_pdb_path + ".dssp"

        try:
            result = subprocess.run(
                [dssp_exec, tmp_pdb_path, tmp_dssp_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0 or not os.path.exists(tmp_dssp_path):
                print(f"Skipped {internal_id}: DSSP failed")
                continue

            with open(tmp_dssp_path, "r") as dssp_f:
                lines = dssp_f.readlines()

            # Find DSSP block
            start = None
            for i, line in enumerate(lines):
                if line.startswith("  #  RESIDUE"):
                    start = i + 1
                    break
            if start is None:
                print(f"Skipped {internal_id}: DSSP output malformed")
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

            # Check strict length match
            if len(ss_list) != len(sequence):
                print(f"Length mismatch: {internal_id} | Seq: {len(sequence)} vs DSSP: {len(ss_list)}")
                continue

            # Write aligned SS/RSA
            out_f.write(f"# {internal_id}\n")
            for ss, rsa in zip(ss_list, rsa_list):
                out_f.write(f"{ss}\t{rsa:.3f}\n")

            written += 1
            if written == 1 or written % 10000 == 0:
                print(f"[Progress] Written {written} proteins")

        finally:
            os.remove(tmp_pdb_path)
            if os.path.exists(tmp_dssp_path):
                os.remove(tmp_dssp_path)

print(f"Done. {written} proteins written to: {output_file}")

