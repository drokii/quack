from typing import List
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from llama_index import PromptTemplate
from quack_prompt_template import quackTemplate
from langchain_core.output_parsers import StrOutputParser

def loadWebPage(webpage : str) -> List[Document]:
    """
    Load content from a web page and extract relevant information using BeautifulSoup.

    Args:
        webpage (str): The URL of the web page to load and extract information from.

    Returns:
        List[Document]: A list of Document objects containing the extracted content.

    Example:
        # Call the function to load a specific web page and extract relevant information
        webpage_url = "https://example.com/sample-page"
        extracted_documents = loadWebPage(webpage_url)
    """

    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(webpage,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print("Quack ate a webpage @ " + webpage)
    return docs

def generateVectorStoreFromPageContents(docs: List[Document]) -> Chroma:

    """
    Chunk the contents of a list of documents into smaller pieces and create a vector store.

    Args:
        docs (List[Document]): A list of documents containing text content to be chunked.

    Returns:
        Chroma: A vector store representing the chunked text content with embeddings.

    Example:
        # Assuming you have a list of Document objects
        document_list = [Document(text="Lorem ipsum..."), Document(text="Another document...")]

        # Call the function to chunk the documents and create a vector store
        vector_store = chunkPageContents(document_list)
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="mistral:latest"))
    print("Quack knows more than ever, as his brain has been enlarged with a vectorstore.")
    return vectorstore


def format_docs(docs : List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)

def promptUsingVectorstore(vectorstore : Chroma, question : str):

    """
    Prompt a question using whatever was added to the vector store.

    Args:
        vectorstore (Chroma): A vector store containing embeddings of text content.
        question (str): The question to ask the language model.

    Example:
        # Call the function to prompt a question using a vector store
        vector_store = loadVectorStore()  # Replace with the actual method to load a vector store
        user_question = "What is the meaning of life?"
        promptUsingVectorstore(vector_store, user_question)
    """

    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model="mistral:latest", temperature=0, request_timeout=15.0)
   
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | quackTemplate()
    | llm
    | StrOutputParser()
    )

    print("Quack has heard your pleas.")

    response = rag_chain.invoke(question)
    return response

def prompt(question : str):
    llm = ChatOllama(model="mistral", temperature=0, request_timeout=15.0)

    response = llm.invoke(question)

    print("Quack has heard your pleas.. \n\n")

    return response


loadedWebPage = loadWebPage("https://lilianweng.github.io/posts/2023-06-23-agent/")
vectorizedPage = generateVectorStoreFromPageContents(loadedWebPage)
modelOutput = promptUsingVectorstore(vectorizedPage, "What is the point of self reflection in AI Models?")

# modelOutput = prompt("Is God dead?")
print(modelOutput)
