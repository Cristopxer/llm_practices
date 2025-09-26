import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

# Prompt Template
prompt = ChatPromptTemplate([
    ("system","You are a helpful assistant. Please response to the question asked"),
    ("user","Question: {question}")
])

# Streamlit
st.title("LangChain Demo with Google Gemma 2b")
input_text = st.text_input("What question do you have in mind?")

# Ollama Llama2 Model
llm = OllamaLLM(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
