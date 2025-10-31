import pandas as pd
from sentence_transformers import SentenceTransformer, util
import glob, os, re

# Load sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(desc1, desc2):
    """Compute cosine similarity between two text descriptions."""
    emb1 = model.encode(desc1, convert_to_tensor=True)
    emb2 = model.encode(desc2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def clean_id(protein_id):
    """Normalize protein IDs: lowercase, strip spaces, remove dashes/underscores."""
    return str(protein_id).strip().lower().replace("-", "").replace("_", "")

if __name__ == "__main__":
    # 1️⃣ Load GO references
    go_df = pd.read_csv("matched_ids_with_go.txt", sep="\t", header=None, names=["protein_id", "go_terms"])
    go_df["protein_id"] = go_df["protein_id"].apply(clean_id)

    # 2️⃣ Read all generated captions
    files = sorted(glob.glob("test_output*_description.txt"))  # sorted ensures order
    data = []

    for i, f in enumerate(files):
        # Map file index to GO ID by order
        if i < len(go_df):
            protein_id = go_df.iloc[i]["protein_id"]
        else:
            protein_id = f"unknown_{i}"  # fallback if more files than GO IDs

        with open(f, "r") as file:
            caption = file.read().strip()

        data.append({"protein_id": protein_id, "generated_caption": caption})

    captions_df = pd.DataFrame(data)

    # 3️⃣ Merge captions with GO references
    merged = pd.merge(captions_df, go_df, on="protein_id", how="inner")

    if merged.empty:
        print("⚠️ No matches found. Check your IDs and formatting.")
    else:
        # 4️⃣ Compute similarity scores
        merged["similarity"] = merged.apply(
            lambda row: similarity_score(row["generated_caption"], row["go_terms"]),
            axis=1
        )

        # 5️⃣ Save results
        merged.to_csv("similarity_results.csv", index=False)
        print(f"✅ Done! {len(merged)} results saved to similarity_results.csv")

    # 6️⃣ Report unmatched counts
    unmatched_captions = set(captions_df["protein_id"]) - set(merged["protein_id"])
    unmatched_go = set(go_df["protein_id"]) - set(merged["protein_id"])
    if unmatched_captions:
        print(f"⚠️ {len(unmatched_captions)} generated captions had no GO match")
    if unmatched_go:
        print(f"⚠️ {len(unmatched_go)} GO IDs had no caption match")
