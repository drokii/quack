from langchain_core.prompts import PromptTemplate

def quackTemplate():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Quack!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    promptTemplate = PromptTemplate.from_template(template)
    return promptTemplate

