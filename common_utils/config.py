import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_llm(tool_mode=False):
    """
    Initialize and return an LLM instance.
    Supports OpenAI, remote vLLM (OpenAI-compatible API), and local Gaudi2 LLM.
    """
    # Check for local LLM first
    use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    local_llm_type = os.getenv("LOCAL_LLM_TYPE", "")
    
    if use_local_llm and local_llm_type == "gaudi2":
        logger.info("✅ Using local Gaudi2 LLM")
        try:
            # Import Gaudi2 LLM wrapper
            from gaudi2_langchain_wrapper import Gaudi2ChatWrapper
            llm = Gaudi2ChatWrapper()
            return llm, False  # Not OpenAI
        except ImportError as e:
            logger.error(f"❌ Failed to import Gaudi2 LLM: {e}")
            logger.info("Falling back to OpenAI/vLLM...")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gaudi2 LLM: {e}")
            logger.info("Falling back to OpenAI/vLLM...")
    
    # Fall back to original OpenAI/vLLM logic
    openai_model = os.getenv("OPENAI_MODEL", "")
    vllm_model = os.getenv("VLLM_MODEL", "")
    llm_model = openai_model or vllm_model

    if not llm_model:
        raise EnvironmentError("No LLM configuration found. Set OPENAI_MODEL, VLLM_MODEL, or USE_LOCAL_LLM=true with LOCAL_LLM_TYPE=gaudi2.")

    # Determine if OpenAI or vLLM
    is_openai = "gpt" in llm_model.lower() or "openai" in llm_model.lower()
    is_vllm = bool(vllm_model) and not is_openai

    if is_openai:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not configured.")
        logger.info("✅ Using OpenAI model: %s", llm_model)
        model_kwargs = {"tool_choice": "auto"} if tool_mode else {}
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0.2,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            request_timeout=60,
            model_kwargs=model_kwargs,
            verbose=True,
        )
    elif is_vllm:
        vllm_api_base = os.getenv("VLLM_API_BASE")
        if not vllm_api_base:
            raise EnvironmentError("VLLM_API_BASE is not configured.")
        logger.info("✅ Using remote vLLM model: %s", llm_model)
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0.2,
            openai_api_key="not-needed",
            openai_api_base=vllm_api_base,
            request_timeout=60,
            model_kwargs={"tool_choice": "auto"} if tool_mode else {},
            verbose=True,
        )
    else:
        raise EnvironmentError("No valid LLM configuration found. Check OPENAI_MODEL, VLLM_MODEL, or local LLM settings.")

    return llm, is_openai
