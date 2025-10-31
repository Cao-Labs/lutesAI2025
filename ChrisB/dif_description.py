import pandas as pd
from sentence_transformers import SentenceTransformer, util
import glob, os, re

# ============================================================
# 🔹 1. Load sentence-transformers model
# ============================================================
model = SentenceTransformer('all-MiniLM-L6-v2')


# ============================================================
# 🔹 2. Cosine similarity function
# ============================================================
def similarity_score(desc1, desc2):
    """Compute cosine similarity between two text descriptions."""
    emb1 = model.encode(desc1, convert_to_tensor=True)
    emb2 = model.encode(desc2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


# ============================================================
# 🔹 3. Helper: clean protein IDs
# ============================================================
def clean_id(protein_id):
    """Normalize protein IDs: lowercase, strip spaces, remove dashes/underscores."""
    return str(protein_id).strip().lower().replace("-", "").replace("_", "")


# ============================================================
# 🔹 4. Load GO OBO file → dictionary {GO_ID: name}
# ============================================================
def load_go_definitions(obo_file):
    go_map = {}
    with open(obo_file, "r") as f:
        current_id, name = None, None
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("name:"):
                name = line.split("name: ")[1]
            elif line == "" and current_id and name:
                go_map[current_id] = name
                current_id, name = None, None
    return go_map


# ============================================================
# 🔹 5. Main pipeline
# ============================================================
if __name__ == "__main__":

    # 1️⃣ Load GO references
    go_df = pd.read_csv("matched_ids_with_go.txt", sep="\t", header=None, names=["protein_id", "go_terms"])
    go_df["protein_id"] = go_df["protein_id"].apply(clean_id)

    # 2️⃣ Load GO definitions from OBO file
    obo_path = "/data/shared/databases/UniProt2025/GO_June_1_2025.obo"
    go_defs = load_go_definitions(obo_path)
    print(f"✅ Loaded {len(go_defs)} GO definitions from {obo_path}")

    # Replace GO IDs with their text definitions when possible
    go_df["go_terms"] = go_df["go_terms"].apply(lambda x: go_defs.get(str(x).strip(), x))

    # 3️⃣ Read all generated BLIP-2 captions
    files = sorted(glob.glob("test_output*_description.txt"))  # sorted ensures consistent ordering
    data = []

    for i, f in enumerate(files):
        if i < len(go_df):
            protein_id = go_df.iloc[i]["protein_id"]
        else:
            protein_id = f"unknown_{i}"  # fallback if more files than GO IDs

        with open(f, "r") as file:
            caption = file.read().strip()

        data.append({"protein_id": protein_id, "generated_caption": caption})

    captions_df = pd.DataFrame(data)

    # 4️⃣ Merge BLIP-2 outputs with GO reference data
    merged = pd.merge(captions_df, go_df, on="protein_id", how="inner")

    if merged.empty:
        print("⚠️ No matches found. Check your IDs and formatting.")
    else:
        # 5️⃣ Compute similarity scores
        merged["similarity"] = merged.apply(
            lambda row: similarity_score(row["generated_caption"], row["go_terms"]),
            axis=1
        )

        # 6️⃣ Save results
        merged.to_csv("similarity_results.csv", index=False)
        print(f"✅ Done! {len(merged)} results saved to similarity_results.csv")

    # 7️⃣ Report unmatched sets
    unmatched_captions = set(captions_df["protein_id"]) - set(merged["protein_id"])
    unmatched_go = set(go_df["protein_id"]) - set(merged["protein_id"])
    if unmatched_captions:
        print(f"⚠️ {len(unmatched_captions)} generated captions had no GO match")
    if unmatched_go:
        print(f"⚠️ {len(unmatched_go)} GO IDs had no caption match")
