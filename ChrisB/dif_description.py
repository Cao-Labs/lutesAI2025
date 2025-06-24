from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
score = util.cos_sim(model.encode(desc1), model.encode(desc2))
