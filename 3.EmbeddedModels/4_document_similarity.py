from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=300
)

documents = [
    "Virat Kohli is a famous Indian cricketer.",
    "Sachin Tendulkar is a legendary Indian cricketer.",
    "Lionel Messi is a famous Argentine footballer.",
    "Bumrah is a fast bowler in the Indian cricket team.",
    "The Indian cricket team won the T20 World Cup."
]

query= "Tell me about Virat Kohli"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

cosine_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

sorted_scores = sorted(
    list(enumerate(cosine_similarities)),
    key=lambda x: x[1],
    reverse=True
)

index, score = sorted_scores[0]

print(query)
print(documents[index])
print("Similarity Score:", score)