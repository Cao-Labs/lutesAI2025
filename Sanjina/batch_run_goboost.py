import os
from pathlib import Path

# Paths
input_dir = Path("/data/summer2020/Boen/benchmark_testing_pdbs")
output_dir = Path("/data/summer2020/Sanjina/goboost_outputs")
config_file = "/data/shared/tools/GOBoost/configs/best/cfg_protein_cc.py"
sub_function = "cc"
prob_threshold = 0.5

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Loop over .pdb files
pdb_files = sorted(input_dir.glob("*.pdb"))

for i, pdb_file in enumerate(pdb_files, 1):
    output_file = output_dir / f"{pdb_file.stem}_goboost.txt"
    print(f"[{i}/{len(pdb_files)}] Processing {pdb_file.name}...")

    cmd = (
        f"python /data/shared/tools/GOBoost/Predictor.py "
        f"--sub_function {sub_function} "
        f"--config-file {config_file} "
        f"--pdb {pdb_file} "
        f"--prob {prob_threshold} > {output_file}"
    )

    os.system(cmd)
