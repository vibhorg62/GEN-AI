from langchain_core.prompts import PromptTemplate

template= PromptTemplate(
        template="""
You are an AI research expert.

Explain the research paper "{paper}".

Explanation Style: {style}

Explanation Length: {length}

Include:
1. Main idea
2. Working
3. Advantages
4. Applications
5. Limitations
""",
        input_variables=["paper", "style", "length"]
    )

template.save('template.json')