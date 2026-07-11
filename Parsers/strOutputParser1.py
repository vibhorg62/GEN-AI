from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Prompt 1
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
template2 = PromptTemplate(
    template="Write a five-line summary of the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model |parser| template2 | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)