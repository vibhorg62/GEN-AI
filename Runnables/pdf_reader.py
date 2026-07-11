from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Embedding model (Free)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Vector Store
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# Retriever
retriever = vectorstore.as_retriever()

# User Query
query = input("Ask Question: ")

# Retrieve documents
retrieved_docs = retriever.invoke(query)

context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
)

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt = f"""
Answer the question only using the context.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)