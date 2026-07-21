from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -------------------- LLM --------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------- Embeddings --------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- Chroma DB --------------------
vector_store = Chroma(
    collection_name="sample",
    persist_directory="my_chroma_db",
    embedding_function=embedding
)

# -------------------- Documents --------------------
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="MS Dhoni is one of the greatest captains in IPL history and led Chennai Super Kings to multiple titles.",
    metadata={"team": "Chennai Super Kings"}
)

doc3 = Document(
    page_content="Jasprit Bumrah is regarded as one of the best fast bowlers in world cricket.",
    metadata={"team": "Mumbai Indians"}
)

doc4 = Document(
    page_content="Andre Russell is a powerful all-rounder who plays for Kolkata Knight Riders.",
    metadata={"team": "Kolkata Knight Riders"}
)

doc5 = Document(
    page_content="Rashid Khan is an exceptional leg-spinner representing Gujarat Titans.",
    metadata={"team": "Gujarat Titans"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# -------------------- Add Documents --------------------
vector_store.add_documents(docs)

print("Documents Added Successfully!")

# -------------------- Similarity Search --------------------
result = vector_store.similarity_search(
    query="Who is a bowler?",
    k=2
)

print("\nSimilarity Search:\n")

for doc in result:
    print(doc.page_content)
    print(doc.metadata)
    print("-"*50)

# -------------------- Similarity Search With Score --------------------
print("\nSimilarity Search With Score:\n")

result = vector_store.similarity_search_with_score(
    query="Who is a bowler?",
    k=1
)

for doc, score in result:
    print("Score:", score)
    print(doc.page_content)
    print("-"*50)

# -------------------- Metadata Filter --------------------
print("\nMetadata Filter:\n")

result = vector_store.similarity_search(
    query="",
    filter={"team": "Chennai Super Kings"}
)

for doc in result:
    print(doc.page_content)

# -------------------- Update Document --------------------
ids = vector_store.get()["ids"]

updated_doc = Document(
    page_content="Virat Kohli is the highest run scorer in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id=ids[0],
    document=updated_doc
)

print("\nDocument Updated!")

# -------------------- Delete Document --------------------
ids = vector_store.get()["ids"]

vector_store.delete(ids=[ids[0]])

print("Document Deleted!")