from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Create Chain
chain = prompt | llm

# User Input
topic = input("Enter a topic: ")

# Run Chain
response = chain.invoke({"topic": topic})

print("Generated Blog Title:")
print(response.content)