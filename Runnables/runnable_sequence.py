from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


#RunnableSequence exist nhi krta abb direct hum chain bna lete hae.. ya toh hume voh use krne ke liye version downgrade krna pdega

load_dotenv()

prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model=ChatGroq(model="llama-3.3-70b-versatile")

parser=StrOutputParser()

prompt2=PromptTemplate(
    template='Explain about this joke {topic}',
    input_variables=['topic']    
)

chain1=prompt1|model|parser
chain2 = prompt1 | model | parser | prompt2 | model | parser



result=chain1.invoke({'topic':'AI'})
final_result=chain2.invoke({'topic':'result'})

print(result)
print(final_result)


#RunnableParallel me bas jaisa maine upar kiya vaisa lekin mergee nhi krna alag alag answer dega.. parallely chalega


#aur jo maine upar kiya voh RunnablePassthrough hae jo chain1 ka result chain2 me pass hua..


# sequnce vala toh ek lambi sequence