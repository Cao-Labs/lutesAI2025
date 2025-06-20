import sys

# ======== CONFIG ========
FEATURES_FILE = "/data/summer2020/naufal/new_features.txt"
TARGET_ID = "COG6_ASPCL"  # <-- Replace with your desired protein ID
# =========================

ss_count = 0
rsa_count = 0
inside_target = False

with open(FEATURES_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            current_id = line[1:].strip()
            inside_target = (current_id == TARGET_ID)
            continue

        if inside_target:
            if line == "" or line.startswith("#"):
                break  # Stop if next protein starts or empty line
            try:
                ss, rsa = line.split("\t")
                ss_count += 1
                rsa_count += 1
            except ValueError:
                print(f"Malformed line: {line}")

# === Output ===
print(f"Protein ID: {TARGET_ID}")
print(f"SS count: {ss_count}")
print(f"RSA count: {rsa_count}")
