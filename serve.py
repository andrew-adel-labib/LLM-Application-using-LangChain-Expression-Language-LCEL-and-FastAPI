import os 
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemma_model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

system_template = "Translate the following into {language} : "
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template), 
    ("user", "{text}")
])

parser = StrOutputParser()
gemma_chain = prompt_template|gemma_model|parser

app = FastAPI(title="Langchain Server",
              version="1.0",
              description="API Server using Langchain Runnable Interfaces")

add_routes(
    app,
    gemma_chain,
    path="/chain"
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




