import pandas as pd
from sentence_transformers import SentenceTransformer, util
import glob

model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(desc1, desc2):
    emb1 = model.encode(desc1, convert_to_tensor=True)
    emb2 = model.encode(desc2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return score.item()

if __name__ == "__main__":
    # 1️⃣ Load GO references
    go_df = pd.read_csv("matched_ids_with_go.txt", sep="\t", header=None, names=["protein_id", "go_terms"])

    # 2️⃣ Read all BLIP-2 text outputs
    files = glob.glob("test_output*_description.txt")  # matches all batch files
    data = []
    for f in files:
        protein_id = f.split("_")[2] if "_" in f else f.split(".")[0]  # extract protein id from filename
        with open(f, "r") as file:
            caption = file.read().strip()
        data.append({"protein_id": protein_id, "generated_caption": caption})

    captions_df = pd.DataFrame(data)

    # 3️⃣ Merge GO references
    merged = pd.merge(captions_df, go_df, on="protein_id", how="inner")

    # 4️⃣ Compute similarity
    similarities = []
    for _, row in merged.iterrows():
        sim = similarity_score(row["generated_caption"], row["go_terms"])
        similarities.append(sim)

    merged["similarity"] = similarities

    # 5️⃣ Save results
    merged.to_csv("similarity_results.csv", index=False)
    print("Done! Results saved to similarity_results.csv")
