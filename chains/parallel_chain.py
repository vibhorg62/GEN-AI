from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=ChatGroq(model="llama-3.3-70b-versatile")
model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt1= PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2= PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)
prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and {quiz}',
    input_variables=['notes','quiz']
)


parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})

merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain

text = """
Artificial Intelligence (AI) is a branch of computer science that enables machines to perform tasks that normally require human intelligence.
Machine learning is a subset of AI that allows systems to learn from data without being explicitly programmed.
Deep learning is a subset of machine learning based on neural networks.
AI is widely used in healthcare, education, finance, transportation, and entertainment.
Virtual assistants like Siri and Google Assistant use AI to understand voice commands.
Recommendation systems on Netflix and YouTube are powered by machine learning algorithms.
AI can improve efficiency, reduce human error, and automate repetitive tasks.
However, AI also raises concerns about privacy, bias, and job displacement.
Ethical AI development focuses on fairness, transparency, and accountability.
The future of AI is expected to bring more intelligent and personalized applications.
"""

result=chain.invoke({'text':text})

print(result)