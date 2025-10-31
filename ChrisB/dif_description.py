import pandas as pd
from sentence_transformers import SentenceTransformer, util
import glob, os, re

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(desc1, desc2):
    """Compute cosine similarity between two texts."""
    emb1 = model.encode(desc1, convert_to_tensor=True)
    emb2 = model.encode(desc2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def clean_id(protein_id):
    """
    Normalize protein IDs:
    - lowercase
    - strip spaces
    - remove dashes and underscores
    """
    return str(protein_id).strip().lower().replace("-", "").replace("_", "")

if __name__ == "__main__":
    # 1️⃣ Load GO references
    go_df = pd.read_csv("matched_ids_with_go.txt", sep="\t", header=None, names=["protein_id", "go_terms"])
    go_df["protein_id"] = go_df["protein_id"].apply(clean_id)

    # 2️⃣ Read all BLIP-2 text outputs
    files = glob.glob("test_output*_description.txt")
    data = []
    for f in files:
        match = re.search(r"test_output(.*?)_description\.txt", os.path.basename(f))
        protein_id = clean_id(match.group(1)) if match else clean_id(os.path.splitext(os.path.basename(f))[0])
        with open(f, "r") as file:
            caption = file.read().strip()
        data.append({"protein_id": protein_id, "generated_caption": caption})

    captions_df = pd.DataFrame(data)
    captions_df["protein_id"] = captions_df["protein_id"].apply(clean_id)

    # 3️⃣ Merge GO references
    merged = pd.merge(captions_df, go_df, on="protein_id", how="inner")

    # 4️⃣ Report unmatched IDs
    unmatched_captions = set(captions_df["protein_id"]) - set(merged["protein_id"])
    unmatched_go = set(go_df["protein_id"]) - set(merged["protein_id"])

    if merged.empty:
        print("⚠️ No matches found. Check your IDs and formatting.")
    else:
        # 5️⃣ Compute similarity
        merged["similarity"] = merged.apply(
            lambda row: similarity_score(row["generated_caption"], row["go_terms"]),
            axis=1
        )

        # 6️⃣ Save results
        merged.to_csv("similarity_results.csv", index=False)
        print(f"✅ Done! {len(merged)} results saved to similarity_results.csv")

    # 7️⃣ Print unmatched info
    if unmatched_captions:
        print(f"⚠️ These generated caption IDs had no GO match ({len(unmatched_captions)}): {unmatched_captions}")
    if unmatched_go:
        print(f"⚠️ These GO IDs had no caption match ({len(unmatched_go)}): {unmatched_go}")
