def process_dssp_file(file_path):
    current_id = None
    current_features = []
    total_ids = 0

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                total_ids += 1
                current_id = line[2:]  # Remove '# ' prefix
                current_features = []  # Reset for new protein
            else:
                current_features.append(line)

    # Print SS/RSA lines for the last protein block
    print("=== SS/RSA for the LAST protein block ===")
    for line in current_features:
        print(line)

    # Print final summary
    print("\n=== Summary ===")
    print(f"Last ID: {current_id}")
    print(f"Total number of protein IDs: {total_ids}")


# Example usage
if __name__ == "__main__":
    file_path = "/data/summer2020/naufal/features_dssp_direct.txt"
    process_dssp_file(file_path)
