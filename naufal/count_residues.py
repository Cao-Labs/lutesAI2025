# count_residues.py

def count_residues(file_path, target_id):
    count = 0
    in_block = False
    target_header = f"# {target_id}"

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if line == target_header:
                    in_block = True
                    continue
                elif in_block:
                    break  # stop when next protein block starts
            elif in_block:
                count += 1

    return count

# Example usage
if __name__ == "__main__":
    file_path = "/data/summer2020/naufal/features_dssp_direct.txt"
    target_id = "COG6_ASPCL"
    result = count_residues(file_path, target_id)
    print(f"Number of residues for {target_id}: {result}")
