import os
from Bio import SeqIO
import subprocess

# Paths
input_fasta = "/data/shared/github/lutesAI2025/Sanjina/test_input/small_test.fasta"
temp_dir = "/data/shared/github/lutesAI2025/Sanjina/temp/"
output_dir = "/data/summer2020/Sanjina/thplm_outputs/"
extract_script = "/data/shared/github/lutesAI2025/Sanjina/THPLM/esmcripts/extract.py"
thplm_script = "/data/shared/github/lutesAI2025/Sanjina/THPLM/THPLM_predict.py"

# Create temp and output dirs
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Start processing
print("Starting batch prediction...")
with open(input_fasta) as handle:
    for i, record in enumerate(SeqIO.parse(handle, "fasta")):
        seq_id = record.id
        sequence = str(record.seq)

        print(f"[{i+1}] Processing {seq_id}")

        # Dynamically create a valid mutation based on actual wild residue
        try:
            wild_residue = sequence[2]  # position 3 (index 2)
        except IndexError:
            print(f"Skipping {seq_id}: sequence too short")
            continue

        if wild_residue != "V":
            variant_placeholder = f"{wild_residue}3V"
        else:
            variant_placeholder = f"{wild_residue}3A"

        # 1. Create variant.txt
        variant_file = os.path.join(temp_dir, "variant.txt")
        with open(variant_file, "w") as vf:
            vf.write(variant_placeholder + "\n")

        # 2. Create fasta file
        wt_fasta = os.path.join(temp_dir, "wild.fasta")
        with open(wt_fasta, "w") as wf:
            wf.write(f">{seq_id}\n{sequence}\n")

        # 3. Create variant_fasta
        mutated_seq = sequence[:2] + variant_placeholder[-1] + sequence[3:]
        variant_fasta = os.path.join(temp_dir, "varlist.fasta")
        with open(variant_fasta, "w") as vfasta:
            vfasta.write(f">{seq_id}\n{sequence}\n")
            vfasta.write(f">{seq_id}_{variant_placeholder}\n{mutated_seq}\n")

        # 4. Create output dir
        sample_output_dir = os.path.join(output_dir, f"{seq_id}/")
        os.makedirs(sample_output_dir, exist_ok=True)

        # 5. Run THPLM
        cmd = [
            "python", thplm_script,
            variant_file,
            wt_fasta,
            sample_output_dir,
            variant_fasta,
            "--gpunumber", "0",
            "--extractfile", extract_script
        ]
        subprocess.run(cmd, check=True)

        # 6. Clean up
        for f in [variant_file, wt_fasta, variant_fasta]:
            os.remove(f)

print("✅ All sequences processed.")
