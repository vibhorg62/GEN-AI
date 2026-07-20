from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
)
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
)

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

parser = StrOutputParser()

# Report Generation
report_gen_chain = prompt1 | model | parser

# Branch
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300,    #kind of if function
        prompt2 | model | parser
    ),
    RunnablePassthrough()
)

# Final Chain
final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({"topic": "Russia vs Ukraine"})

print(result)