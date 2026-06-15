from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["A vector database stores embeddings.", "It is used for similarity search."]
embeddings = model.encode(sentences)
score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity: {round(score, 4)}")