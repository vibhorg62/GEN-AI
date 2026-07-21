from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --------------------- LLM ---------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# --------------------- Embedding ---------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------- Documents ---------------------

docs = [

    Document(
        page_content="Virat Kohli is one of the greatest IPL batsmen.",
        metadata={"team":"RCB"}
    ),

    Document(
        page_content="MS Dhoni is the captain of Chennai Super Kings.",
        metadata={"team":"CSK"}
    ),

    Document(
        page_content="Jasprit Bumrah is India's best fast bowler.",
        metadata={"team":"MI"}
    ),

    Document(
        page_content="Andre Russell is an explosive all-rounder.",
        metadata={"team":"KKR"}
    ),

    Document(
        page_content="Rashid Khan is a world class leg spinner.",
        metadata={"team":"GT"}
    )
]

# --------------------- Create FAISS DB ---------------------

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

print("FAISS Index Created Successfully!")

# --------------------- Save ---------------------

vector_store.save_local("faiss_index")

print("Index Saved!")

# --------------------- Load ---------------------

vector_store = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

print("Index Loaded!")

# --------------------- Similarity Search ---------------------

results = vector_store.similarity_search(
    "Who is a bowler?",
    k=2
)

print("\nSimilarity Search\n")

for doc in results:

    print(doc.page_content)
    print(doc.metadata)
    print("---------------------")

# --------------------- Similarity Search with Score ---------------------

results = vector_store.similarity_search_with_score(
    "Who is a bowler?",
    k=2
)

print("\nSimilarity Search With Score\n")

for doc, score in results:

    print(score)
    print(doc.page_content)
    print("---------------------")

# --------------------- Metadata Filter ---------------------

results = vector_store.similarity_search(
    "captain",
    k=2
)

print("\nMetadata Filter\n")

for doc in results:

    if doc.metadata["team"] == "CSK":
        print(doc.page_content)
