#Data Ingestion
from langchain_community.document_loaders import TextLoader
loader = TextLoader("speech.txt")
documents = loader.load()

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["USER_AGENT"] = "simplerag-app/1.0"

# Web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4

#Load, chunk & index the content of html document
web_loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                               class_=("post-title","post-content","post-header","post-footer")
                           )),)

documents = web_loader.load()
#print(documents)

from langchain_community.document_loaders import PyPDFLoader
# Load, chunk & index the content of pdf document
pdf_loader = PyPDFLoader("attention.pdf")
documents = pdf_loader.load()
#print(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print("Iam here")
from langchain_cohere import CohereEmbeddings
# Create embeddings for the text chunks
embeddings = CohereEmbeddings(
    model="embed-english-v2.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)
from langchain_community.vectorstores import FAISS
# Create a vector store from the text chunks and embeddings
vector_store = FAISS.from_documents(texts, embeddings)
query = "Who are the authors of attention is all you need?"
retrieved_results=vector_store.similarity_search(query)
#print("hi:",retrieved_results[0].page_content)
#print("12345")


from langchain_community.llms import Cohere
llm = Cohere()

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following Question based on the context.
    Think step by step before  answering the question.
    I will give you treat.
    <context>
    {context}
    </context>
    Question: {question}
    """
)


