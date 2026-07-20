from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from dotenv import load_dotenv

load_dotenv()


def word_count(text):
    return len(text.split())


prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

parser = StrOutputParser()

# Joke generation chain
joke_gen_chain = prompt | model | parser

# Parallel chain
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count),
    }
)

# Final chain
final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({"topic": "AI"})

print(result["joke"])
print("Word Count:", result["word_count"])