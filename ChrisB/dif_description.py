from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def similarity_score(desc1, desc2):
    emb1 = model.encode(desc1, convert_to_tensor=True)
    emb2 = model.encode(desc2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2)
    return score.item()

# Example test run (can remove in production)
if __name__ == "__main__":
    desc_generated = "Protein acts as a transporter of ions."
    desc_reference = "This protein functions in ion transport across membranes."
    print("Similarity:", similarity_score(desc_generated, desc_reference))
