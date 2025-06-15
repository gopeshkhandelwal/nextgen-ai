import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
    return ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
        max_tokens=128,
        temperature=0
    )


def get_openweather_api_key():
    return os.getenv("OPENWEATHER_API_KEY")
