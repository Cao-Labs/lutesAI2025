# seq_len.py

FASTA_FILE = "/data/summer2020/naufal/training_data/protein_sequences.fasta"
TARGET_ID = "104K_THEPA"  # Replace with the ID you want to check

def get_sequence_length(fasta_path, target_id):
    with open(fasta_path, "r") as f:
        current_id = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id == target_id:
                    return len("".join(seq))
                current_id = line[1:]
                seq = []
            else:
                seq.append(line)
        # Final check in case target is last in file
        if current_id == target_id:
            return len("".join(seq))
    return None

length = get_sequence_length(FASTA_FILE, TARGET_ID)
if length is not None:
    print(f"Sequence length of {TARGET_ID}: {length}")
else:
    print(f"Protein ID {TARGET_ID} not found in FASTA file.")
