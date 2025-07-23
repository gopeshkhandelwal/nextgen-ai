import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Optimum Habana REST API wrapper for LangChain
class OptimumHabanaLLM:
    def __init__(self, api_base, model, max_tokens=4096, temperature=0.7):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.bound_tools = []

    def generate(self, prompt, **kwargs):
        # Remove the aggressive truncation - let the full prompt through
        # max_prompt_length = 300  # Remove this line
        # if len(prompt) > max_prompt_length:  # Remove this block
        #     prompt = prompt[:max_prompt_length] + "..."
            
        payload = {
            "model": self.model,
            "prompt": prompt,  # Send full prompt
            "max_tokens": min(kwargs.get("max_tokens", self.max_tokens), 256),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        print(f"DEBUG: Sending request to {self.api_base}/generate")
        print(f"DEBUG: Payload: {payload}")
        
        try:
            import time
            start_time = time.time()
            
            # Add headers explicitly
            headers = {"Content-Type": "application/json"}
            print(f"DEBUG: Headers: {headers}")
            
            resp = requests.post(
                f"{self.api_base}/generate", 
                json=payload, 
                headers=headers,
                timeout=180
            )
            
            elapsed = time.time() - start_time
            print(f"DEBUG: Request completed in {elapsed:.2f}s")
            print(f"DEBUG: Response status: {resp.status_code}")
            print(f"DEBUG: Response headers: {dict(resp.headers)}")
            
            if resp.status_code != 200:
                print(f"DEBUG: Response text: {resp.text}")
            
            resp.raise_for_status()
            response_json = resp.json()
            print(f"DEBUG: Response JSON: {response_json}")
            
            # Try both 'response' and 'text' fields for compatibility
            return response_json.get("response", response_json.get("text", ""))
            
        except requests.exceptions.Timeout:
            print("DEBUG: Request timed out after 180s")
            return "[Connection timeout - server taking longer than 180s]"
        except requests.exceptions.ConnectionError as e:
            print(f"DEBUG: Connection error: {e}")
            return "[Connection failed - server unreachable]"
        except requests.exceptions.HTTPError as e:
            print(f"DEBUG: HTTP error: {e}")
            return f"[HTTP error: {e}]"
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Request error: {e}")
            return f"[Request error: {e}]"
        except Exception as e:
            print(f"DEBUG: Unexpected error: {e}")
            return f"[Unexpected error: {e}]"

    def bind_tools(self, tools):
        """Store tools for compatibility with LangChain, but don't use them directly"""
        self.bound_tools = tools
        return self

    def invoke(self, messages, **kwargs):
        """LangChain-compatible invoke method optimized for Meta-Llama-3.1-8B-Instruct"""
        from langchain_core.messages import AIMessage
        
        print(f"DEBUG: Raw messages received: {messages}")  # Debug line
        
        # Build the complete prompt with system instructions AND user question
        system_content = ""
        user_question = ""
        
        # Process messages to extract system and user content
        for msg in messages:
            print(f"DEBUG: Processing message: {type(msg)}")
            
            content = None
            msg_type = None
            
            if isinstance(msg, dict):
                content = msg.get('content', '')
                msg_type = msg.get('role', '')
                print(f"DEBUG: Dict message - type: {msg_type}, content: '{content[:50]}...'")
            elif hasattr(msg, 'content'):
                content = msg.content
                msg_type = getattr(msg, 'type', getattr(msg, '__class__', '').__name__.lower())
                print(f"DEBUG: Object message - type: {msg_type}, content: '{content[:50]}...'")
            
            if content and content.strip():
                content = content.strip()
                
                if msg_type == 'system':
                    system_content = content
                    print(f"DEBUG: Found system instructions: '{content[:100]}...'")
                elif msg_type in ['human', 'user'] or msg_type == 'user':
                    user_question = content
                    print(f"DEBUG: Found user question: '{user_question}'")
        
        # Build complete prompt with system instructions + user question
        if system_content and user_question:
            # Format as instruction-following prompt for Llama-3.1
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            print(f"DEBUG: Built complete prompt with system instructions and user question")
        elif user_question:
            prompt = user_question
            print(f"DEBUG: Using user question only (no system instructions)")
        else:
            prompt = "Hello"
            print(f"DEBUG: Fallback to default prompt")
        
        print(f"DEBUG: Final prompt being sent to Meta-Llama-3.1-8B-Instruct: '{prompt[:200]}...'")
        
        # Generate response using Meta-Llama-3.1-8B-Instruct
        response_text = self.generate(prompt, **kwargs)
        
        # Return as AIMessage
        return AIMessage(content=response_text)

    async def ainvoke(self, messages, **kwargs):
        """LangChain-compatible async invoke method"""
        # For now, just call the sync version since our REST API is blocking anyway
        return self.invoke(messages, **kwargs)

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

def get_llm(tool_mode=False):
    """
    Initialize and return an LLM instance.
    Supports OpenAI, remote vLLM, and Optimum Habana (REST API).
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model = os.getenv("OPENAI_MODEL", "")
    vllm_model = os.getenv("VLLM_MODEL", "")
    optimum_model = os.getenv("OPTIMUM_HABANA_MODEL", "")

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
            openai_api_key="not-needed",
            openai_api_base=vllm_api_base,
            request_timeout=60,
            model_kwargs=model_kwargs,
            verbose=True,
        )
        return llm, False
    elif provider == "optimum_habana":
        optimum_api_base = os.getenv("OPTIMUM_HABANA_API_BASE")
        if not optimum_model or not optimum_api_base:
            raise EnvironmentError("OPTIMUM_HABANA_MODEL or OPTIMUM_HABANA_API_BASE is not configured.")
        logger.info(f"✅ Using Optimum Habana model: {optimum_model} at {optimum_api_base}")
        llm = OptimumHabanaLLM(
            api_base=optimum_api_base,
            model=optimum_model,
            max_tokens=os.getenv("OPTIMUM_HABANA_MAX_TOKENS", 4096),
            temperature=os.getenv("OPTIMUM_HABANA_TEMPERATURE", 0.7),
        )
        return llm, False
    else:
        raise EnvironmentError(f"LLM_PROVIDER '{provider}' is not supported.")
