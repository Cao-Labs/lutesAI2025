import os
from Bio import SeqIO
import subprocess

# Paths
input_fasta = "/data/summer2020/Boen/benchmark_testing_sequences.fasta"
variant_placeholder = "A3V"  # Dummy variant for now
temp_dir = "/data/shared/github/lutesAI2025/Sanjina/temp/"
output_dir = "/data/summer2020/Sanjina/thplm_outputs/"
extract_script = "/data/shared/tools/THPLM/esmcripts/extract.py"
thplm_script = "/data/shared/tools/THPLM/THPLM_Predict.py"

# Create temp and output dirs
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Output results dict
results_file = os.path.join(output_dir, "thplm_ddg_results.json")

# Start processing
print("Starting batch prediction...")
with open(input_fasta) as handle:
    for i, record in enumerate(SeqIO.parse(handle, "fasta")):
        seq_id = record.id
        sequence = str(record.seq)

        print(f"[{i+1}] Processing {seq_id}")

        # 1. Create variant.txt
        variant_file = os.path.join(temp_dir, "variant.txt")
        with open(variant_file, "w") as vf:
            vf.write(variant_placeholder + "\n")  # You can customize this

        # 2. Create fasta file
        wt_fasta = os.path.join(temp_dir, "wild.fasta")
        with open(wt_fasta, "w") as wf:
            wf.write(f">{seq_id}\n{sequence}\n")

        # 3. Create variant_fasta with dummy mutated sequence
        variant_fasta = os.path.join(temp_dir, "varlist.fasta")
        mutated_seq = sequence[:2] + "V" + sequence[3:]  # Just for testing
        with open(variant_fasta, "w") as vfasta:
            vfasta.write(f">{seq_id}\n{sequence}\n")
            vfasta.write(f">{seq_id}_{variant_placeholder}\n{mutated_seq}\n")

        # 4. Create sub-output directory
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
        subprocess.run(cmd)

        # 6. Clean up temp files
        for f in [variant_file, wt_fasta, variant_fasta]:
            os.remove(f)

print("✅ All sequences processed.")
