from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Cleaned_Marketing_Data.csv')

docs = loader.load()

print(len(docs))
print(docs[1])