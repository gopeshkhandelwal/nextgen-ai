from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    """Get the LLM instance for the RAG pipeline."""
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )