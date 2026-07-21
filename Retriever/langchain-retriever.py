#<-------WIKIPEDIA RETRIEVER---------->

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=1,
    lang="en",
    load_all_available_meta=False,
    doc_content_chars_max=1000
)

query = "the geopolitical history of india and pakistan from the perspective of a chinese"
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  
