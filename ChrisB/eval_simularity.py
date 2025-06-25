import os
from dif_description import similarity_score
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 1: Parse GO .obo file ---
def parse_go_obo(obo_path):
    go_terms = {}
    with open(obo_path, 'r', encoding='utf-8') as f:
        term = {}
        for line in f:
            line = line.strip()
            if line == "[Term]":
                term = {}
            elif line.startswith("id: GO:"):
                term["id"] = line.split("id: ")[1]
            elif line.startswith("name: "):
                term["name"] = line.split("name: ")[1]
            elif line.startswith('def: "'):
                term["def"] = line.split('def: "')[1].split('"')[0]
            elif line == "" and "id" in term and "def" in term:
                go_terms[term["id"]] = {
                    "name": term.get("name", ""),
                    "definition": term["def"]
                }
    return go_terms

# --- Step 2: Load filtered GO IDs ---
def load_matched_ids(filepath):
    with open(filepath, "r") as f:
        return {line.strip() for line in f if line.startswith("GO:")}

# --- Step 3: Compute top-N most similar GO definitions ---
def get_top_go_matches(generated_desc, reference_go_dict, top_k=5):
    gen_embedding = model.encode(generated_desc, convert_to_tensor=True)
    results = []

    for go_id, entry in reference_go_dict.items():
        ref_embedding = model.encode(entry["definition"], convert_to_tensor=True)
        score = util.cos_sim(gen_embedding, ref_embedding).item()
        results.append((go_id, entry["name"], entry["definition"], score))

    results.sort(key=lambda x: x[3], reverse=True)  # sort by similarity
    return results[:top_k]

# --- Main execution ---
if __name__ == "__main__":
    # Step 4: Get your input description
    generated_description = input("Enter BLIP-2 generated description:\n> ").strip()

    # Step 5: Load data
    go_path = "GO_June_1_2025.obo"
    match_path = "matched_ids_with_go.txt"
    print("\n[+] Loading GO terms and matched IDs...")
    all_go_terms = parse_go_obo(go_path)
    matched_ids = load_matched_ids(match_path)
    filtered_go = {go_id: all_go_terms[go_id] for go_id in matched_ids if go_id in all_go_terms}

    # Step 6: Compute top matches
    print("\n[+] Computing similarity...")
    top_matches = get_top_go_matches(generated_description, filtered_go, top_k=5)

    print("\n🔍 Top Matching GO Terms:")
    for go_id, name, definition, score in top_matches:
        print(f"\nGO ID: {go_id}")
        print(f"Name: {name}")
        print(f"Definition: {definition}")
        print(f"Similarity Score: {score:.4f}")
