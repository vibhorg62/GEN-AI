from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

parser=JsonOutputParser()

template=PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt=template.format()

# result = model.invoke(prompt)     #instead of this we can make a chain

chain = template | model | parser
result = chain.invoke({'topic':'black Hole'})

print(result) 


# final_result=parser.parse(result.content)

# print(final_result)