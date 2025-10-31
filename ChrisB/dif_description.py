import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(desc1, desc2):
    emb1 = model.encode(desc1, convert_to_tensor=True)
    emb2 = model.encode(desc2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return score.item()

if __name__ == "__main__":
    # 1️⃣ Load GO references
    go_df = pd.read_csv("matched_ids_with_go.txt", sep="\t", header=None, names=["protein_id", "go_terms"])

    # 2️⃣ Load generated captions (update filename accordingly)
    captions_df = pd.read_csv("generated_captions.csv")  # assume columns: protein_id, generated_caption

    # 3️⃣ Merge both on protein_id
    merged = pd.merge(captions_df, go_df, on="protein_id")

    # 4️⃣ Compute similarity
    similarities = []
    for _, row in merged.iterrows():
        sim = similarity_score(row["generated_caption"], row["go_terms"])
        similarities.append(sim)

    merged["similarity"] = similarities

    # 5️⃣ Save results
    merged.to_csv("similarity_results.csv", index=False)
    print("Done! Results saved to similarity_results.csv")
