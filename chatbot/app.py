from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import Cohere
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

## Langsmith Tracking

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Set page config
st.set_page_config(page_title="Cohere Chatbot", page_icon="💬")

st.title("💬 Cohere Chatbot with LangChain")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title('Langchain Demo With Cohere API')
input_text=st.text_input("Search the topic u want")

# cohere LLm 
llm=Cohere()
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
