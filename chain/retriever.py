from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Cohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load & Split
docs = PyPDFLoader("attention.pdf").load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(docs)

# 2. Build Vector Store
embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

# 3. LLM & Prompt
llm = Cohere()
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following Question based on the context.
    Think step by step before answering.
    <context>
    {context}
    </context>
    Question: {question}
    """
)

# 4. Chains & Retrieval
doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

# 5. Query
resp = retrieval_chain.invoke({"question": "Who are the authors of attention is all you need?"})
print("Answer:", resp["answer"])
