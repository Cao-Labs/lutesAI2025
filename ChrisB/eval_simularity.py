import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 🔹 1. Load sentence-transformers model
# ============================================================
model = SentenceTransformer('all-MiniLM-L6-v2')


# ============================================================
# 🔹 2. Parse GO .obo file into dict {GO_ID: {name, definition}}
# ============================================================
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


# ============================================================
# 🔹 3. Load list of matched GO IDs
# ============================================================
def load_matched_ids(filepath):
    with open(filepath, "r") as f:
        return {line.strip() for line in f if line.startswith("GO:")}


# ============================================================
# 🔹 4. Compute top-N most similar GO definitions for a caption
# ============================================================
def get_top_go_matches(generated_desc, reference_go_dict, top_k=5):
    gen_embedding = model.encode(generated_desc, convert_to_tensor=True)
    results = []
    for go_id, entry in reference_go_dict.items():
        ref_embedding = model.encode(entry["definition"], convert_to_tensor=True)
        score = util.cos_sim(gen_embedding, ref_embedding).item()
        results.append((go_id, entry["name"], entry["definition"], score))
    results.sort(key=lambda x: x[3], reverse=True)
    return results[:top_k]


# ============================================================
# 🔹 5. Main: evaluate all BLIP-2 captions automatically
# ============================================================
if __name__ == "__main__":
    go_path = "GO_June_1_2025.obo"
    match_path = "matched_ids_with_go.txt"
    print("\n[+] Loading GO terms and matched IDs...")
    all_go_terms = parse_go_obo(go_path)
    matched_ids = load_matched_ids(match_path)
    filtered_go = {go_id: all_go_terms[go_id] for go_id in matched_ids if go_id in all_go_terms}
    print(f"✅ Loaded {len(filtered_go)} filtered GO terms.")

    # Find all caption files
    caption_files = sorted(glob.glob("test_output*_description.txt"))
    if not caption_files:
        print("⚠️ No caption files found (expected test_output*_description.txt)")
        exit()

    all_results = []

    for fpath in caption_files:
        with open(fpath, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        base = os.path.basename(fpath)
        print(f"\n🔍 Evaluating {base} ...")

        top_matches = get_top_go_matches(caption, filtered_go, top_k=5)

        for go_id, name, definition, score in top_matches:
            all_results.append({
                "file": base,
                "generated_caption": caption,
                "go_id": go_id,
                "go_name": name,
                "go_definition": definition,
                "similarity_score": score
            })

    # Save combined results
    out_path = "top_similarity_results.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\n✅ Done! Saved top-5 GO matches per caption → {out_path}")
