import os
import subprocess
from tempfile import NamedTemporaryFile

# === File paths ===
fasta_path = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
idmap_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"
pdb_dir = "/data/shared/databases/alphaFold"
dssp_exec = "/data/shared/tools/DeepQA/tools/dsspcmbi"
output_file = "/data/summer2020/naufal/features_dssp_direct.txt"

# === Read all internal IDs from FASTA ===
print("Indexing FASTA IDs...")
fasta_ids = []
with open(fasta_path, "r") as f:
    for line in f:
        if line.startswith(">"):
            fasta_ids.append(line[1:].strip())
print(f"Found {len(fasta_ids)} sequence IDs.\n")

# === Open output ===
with open(output_file, "w") as out_f:

    processed = 0
    matched = set()

    # === Stream ID mapping file ===
    with open(idmap_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            accession, internal_id = parts
            if internal_id not in fasta_ids or internal_id in matched:
                continue

            matched.add(internal_id)
            pdb_path = os.path.join(pdb_dir, f"AF-{accession}-F1-model_v4.pdb")
            if not os.path.exists(pdb_path):
                print(f"Missing PDB for {internal_id}")
                continue

            # === Run DSSP ===
            with open(pdb_path, "r") as pdb_file:
                pdb_lines = pdb_file.readlines()

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
                    continue

                with open(tmp_dssp_path, "r") as dssp_f:
                    lines = dssp_f.readlines()

                start = None
                for i, line in enumerate(lines):
                    if line.startswith("  #  RESIDUE"):
                        start = i + 1
                        break
                if start is None:
                    print(f"Skipped {internal_id}: Malformed DSSP")
                    continue

                out_f.write(f"# {internal_id}\n")
                for line in lines[start:]:
                    if len(line) < 38:
                        continue
                    try:
                        ss = line[16].strip()
                        ss = ss if ss in ("H", "E") else "C"
                        rsa = float(line[35:38].strip())
                        out_f.write(f"{ss}\t{rsa:.3f}\n")
                    except:
                        continue

                processed += 1
                if processed == 1 or processed % 10000 == 0:
                    print(f"Processed {processed} proteins...")

            finally:
                os.remove(tmp_pdb_path)
                if os.path.exists(tmp_dssp_path):
                    os.remove(tmp_dssp_path)

print(f"\nDone. DSSP features written to: {output_file}")

