# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_llm(tool_mode=False):
    """
    Initialize and return an LLM instance.
    Supports OpenAI and remote vLLM only.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model = os.getenv("OPENAI_MODEL", "")
    vllm_model = os.getenv("VLLM_MODEL", "")

    if provider == "openai":
        if not openai_model:
            raise EnvironmentError("OPENAI_MODEL is not configured.")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not configured.")
        logger.info("✅ Using OpenAI model: %s", openai_model)
        # Only set tool_choice when explicitly in tool mode
        model_kwargs = {}
        if tool_mode:
            model_kwargs["tool_choice"] = "auto"
        llm = ChatOpenAI(
            model=openai_model,
            temperature=0.2,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            request_timeout=60,
            model_kwargs=model_kwargs,
            verbose=True,
        )
        return llm, True
    elif provider == "vllm":
        if not vllm_model:
            raise EnvironmentError("VLLM_MODEL is not configured.")
        vllm_api_base = os.getenv("VLLM_API_BASE")
        if not vllm_api_base:
            raise EnvironmentError("VLLM_API_BASE is not configured.")
        logger.info("✅ Using remote vLLM model: %s", vllm_model)
        model_kwargs = {}
        if tool_mode:
            model_kwargs["tool_choice"] = "auto"
        llm = ChatOpenAI(
            model=vllm_model,
            temperature=0.2,
            api_key="not-needed",
            base_url=vllm_api_base,
            request_timeout=60,
            model_kwargs=model_kwargs,
            verbose=True,
        )
        return llm, False
    else:
        raise EnvironmentError(f"LLM_PROVIDER '{provider}' is not supported. Use 'openai' or 'vllm'.")
