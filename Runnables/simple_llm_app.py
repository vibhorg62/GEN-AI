from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

prompt=PromptTemplate(
    input_variables=['topic'],
    template='Suggest a Catchy blog title about {topic}'
)

topic=input('Enter a topic : ')

formatted_prompt=prompt.format(topic=topic)

blog_title=llm.invoke(formatted_prompt)

print("Generated Blog Title : ",blog_title)