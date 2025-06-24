import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_llm():
    """
    Initialize and return a ChatOpenAI LLM instance for the RAG pipeline.
    Logs and raises an error if required environment variables are missing.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment.")
        raise EnvironmentError("OPENAI_API_KEY is not configured.")

    openai_model = os.getenv("OPENAI_MODEL")
    if not openai_model:
        logger.error("OPENAI_MODEL not found in environment.")
        raise EnvironmentError("OPENAI_MODEL is not configured.")

    logger.info("Initializing ChatOpenAI with model: %s", openai_model)
    return ChatOpenAI(
        model=openai_model,
        temperature=0.2,
        openai_api_key=openai_api_key,
        request_timeout=60,
        model_kwargs={
            "tool_choice": "auto"
        }
        verbose=True
    )