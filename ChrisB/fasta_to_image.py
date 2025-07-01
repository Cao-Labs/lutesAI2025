import sys
import subprocess

def read_fasta(filepath):
    sequences = []
    with open(filepath, 'r') as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)
    return sequences

fasta_file = sys.argv[1]
sequences = read_fasta(fasta_file)

for i, seq in enumerate(sequences):
    output_name = f"protein_{i+1}_image.png"
    cmd = f'python generate_protein_image.py --sequence "{seq}" --out {output_name}'
    subprocess.run(cmd, shell=True)
    print(f"Generated {output_name}")