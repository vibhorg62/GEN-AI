from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple words what is {topic} and how it works.')
])

prompt=chat_template.invoke({
    "domain": "cricket",
    "topic": "T20"
})

print(prompt)