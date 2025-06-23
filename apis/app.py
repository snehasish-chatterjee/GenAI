import uvicorn
from fastapi import FastAPI
import os
from dotenv import load_dotenv
from langchain_community.llms import Cohere
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes

from pydantic import BaseModel

class EssayInput(BaseModel):
    topic: str
EssayInput.model_rebuild()

load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"

)

llm=Cohere()
prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
chain=prompt1 | llm

add_routes(
    app,
    chain.with_types(input_type=EssayInput),
    path="/essay",
    enabled_endpoints=["invoke", "config_hashes"],  # ✅ only include /essay/invoke
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)