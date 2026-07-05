from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

documents=[
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Germany is Berlin."
]

vectors = embeddings.embed_documents(documents)

print(str(vectors))