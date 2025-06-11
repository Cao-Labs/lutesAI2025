# generate_idmapping_uni_minimal.py
# This file extracts relevant information from the huge id mapping file

input_file = "/data/shared/databases/UniProt2025/idmapping.dat"
output_file = "/data/shared/databases/UniProt2025/idmapping_uni.txt"

def extract_uniprotkb_id(input_path, output_path):
    seen = set()  # to avoid writing the same original ID twice

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            parts = line.strip().split("\t")
            if len(parts) == 3 and parts[1] == "UniProtKB-ID":
                original_id = parts[0]
                uniprotkb_id = parts[2]
                if original_id not in seen:
                    outfile.write(f"{original_id}\t{uniprotkb_id}\n")
                    seen.add(original_id)

if __name__ == "__main__":
    extract_uniprotkb_id(input_file, output_file)
