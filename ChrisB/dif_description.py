import pandas as pd
from sentence_transformers import SentenceTransformer, util
import glob
import os
import re

model = SentenceTransformer('all‑MiniLM‑L6‑v2')

def similarity_score(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def clean_id(protein_id):
    """Normalize protein IDs: lowercase, strip spaces, remove dashes/underscores."""
    return str(protein_id).strip().lower().replace("-", "").replace("_", "")

def load_go_descriptions(go_ids):
    """
    Given a list of GO IDs (e.g., ["GO:0005524", "GO:0003677"]), return a
    dictionary mapping GO ID → description (term name or definition).
    Requires you to have or download a GO term table (ID → name/definition).
    """
    # Example minimal implementation: load a file `go_terms.tsv` with columns: GO_ID, Name
    go_table = pd.read_csv("go_terms.tsv", sep="\t", dtype=str)  # adjust path/format
    go_table = go_table.set_index("GO_ID")["Name"].to_dict()
    descs = {}
    for gid in go_ids:
        descs[gid] = go_table.get(gid, "")
    return descs

if __name__ == "__main__":
    # 1. Load GO references (your protein→GO mapping)
    go_df = pd.read_csv("matched_ids_with_go.txt", sep="\t", header=None,
                        names=["protein_id", "go_terms"], dtype=str)
    go_df["protein_id"] = go_df["protein_id"].apply(clean_id)

    # 2. Read all generated caption files
    files = glob.glob("test_output*_description.txt")
    data = []
    for f in files:
        base = os.path.basename(f)
        match = re.search(r"test_output(.*?)_description\.txt", base)
        if match:
            pid = match.group(1)
        else:
            pid = os.path.splitext(base)[0]
        pid_clean = clean_id(pid)
        with open(f, "r", encoding="utf‑8") as file:
            caption = file.read().strip()
        data.append({"protein_id": pid_clean, "generated_caption": caption})

    captions_df = pd.DataFrame(data)
    captions_df["protein_id"] = captions_df["protein_id"].apply(clean_id)

    # 3. Merge GO references with caption data
    merged = pd.merge(captions_df, go_df, on="protein_id", how="inner")

    # 4. Report unmatched IDs
    unmatched_captions = set(captions_df["protein_id"]) - set(merged["protein_id"])
    unmatched_go = set(go_df["protein_id"]) - set(merged["protein_id"])

    if merged.empty:
        print("⚠️ No matches found. Check your IDs and formatting.")
        print("Unmatched caption IDs:", unmatched_captions)
        print("Unmatched GO mapping IDs:", unmatched_go)
        exit(1)

    # 5. Convert GO ID strings to descriptions
    #    e.g., "GO:0005524;GO:0003677" → list of ids → map to term names → join into description string
    def map_go_to_text(go_ids_str):
        go_ids = [gid.strip() for gid in go_ids_str.split(";") if gid.strip()]
        desc_map = load_go_descriptions(go_ids)
        # take the names and join them into one text blob
        descs = [desc_map.get(gid, "") for gid in go_ids if desc_map.get(gid, "")]
        if not descs:
            return ""
        return " ; ".join(descs)

    merged["go_description"] = merged["go_terms"].apply(map_go_to_text)

    # 6. Compute similarity between generated caption and GO description
    merged["similarity"] = merged.apply(
        lambda row: similarity_score(row["generated_caption"], row["go_description"]),
        axis=1
    )

    # 7. Save results
    merged.to_csv("similarity_results_with_go_desc.csv", index=False)
    print(f"✅ Done! {len(merged)} results saved to similarity_results_with_go_desc.csv")

    # 8. Print unmatched info
    if unmatched_captions:
        print(f"⚠️ These generated caption IDs had no GO match ({len(unmatched_captions)}): {unmatched_captions}")
    if unmatched_go:
        print(f"⚠️ These GO IDs had no caption match ({len(unmatched_go)}): {unmatched_go}")
