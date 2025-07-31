import os
import logging
import uuid
import time
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt.tool_node import ToolNode
from langgraph_tool_wrappers import build_tool_wrappers
from langchain_openai import ChatOpenAI
from db_utils import get_last_n_messages
from common_utils.config import get_llm
from common_utils.utils import to_openai_dict, sanitize_message_history
import json

# Langfuse integration
try:
    from langfuse.callback import CallbackHandler
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("⚠️ Langfuse not installed. Run: pip install langfuse")


# Load environment variables from .env file
load_dotenv()

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize Langfuse if available and enabled
langfuse_handler = None
if LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_ENABLED", "false").lower() == "true":
    try:
        langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        )
        logger.info("✅ Langfuse monitoring enabled")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Langfuse: {e}")
        langfuse_handler = None
else:
    logger.info("🔍 Langfuse monitoring disabled")

class AgentState(TypedDict, total=False):
    """
    Defines the state structure for the agent, including messages and user/session identifiers.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    output: str
    user_id: str
    session_id: str
    trace_id: str  # For E2E tracing
    start_time: float  # For performance monitoring
    
# Function to check if the LLM response is low-confidence
async def is_low_confidence(content: str) -> bool:
    """
    Check if the response contains any low-confidence phrases.
    These phrases are configurable via the LOW_CONFIDENCE_PHRASES env variable.
    Uses substring matching for more flexible detection.
    """
    if not content:
        return False
        
    phrases = os.getenv("LOW_CONFIDENCE_PHRASES", "")
    low_conf_phrases = [p.strip().lower() for p in phrases.split(",") if p.strip()]
    
    content_lower = content.lower()
    for phrase in low_conf_phrases:
        if phrase in content_lower:
            logger.info(f"🔍 Low-confidence detected: found '{phrase}' in response")
            return True
    
    return False

# Load tools
tools = build_tool_wrappers()

# Initialize LLM + bind tools if OpenAI
llm, _ = get_llm(tool_mode=True)
llm_with_tools = llm.bind_tools(tools)

async def router(state: AgentState) -> AgentState:
    """
    Router node for the agent:
    - Uses short-term memory by default.    
    - Automatically retries with long-term memory if LLM response is low-confidence.
    """
    # Initialize tracing for this request
    if "trace_id" not in state:
        state["trace_id"] = str(uuid.uuid4())
        state["start_time"] = time.time()
        logger.info(f"🎯 Starting E2E trace: {state['trace_id']}")

    trace_id = state["trace_id"]
    
    system_message = SystemMessage(
        content="""You are a helpful assistant. Use the tools when needed, but BE SELECTIVE - only call tools that are directly relevant to the user's question.

        TOOL SELECTION GUIDELINES:
        - For weather questions: ONLY use city_weather tool
        - For IDC gRPC API questions: ONLY use document_qa tool  
        - For IDC pools/images: ONLY use list_idc_pools or machine_images
        - For ITAC products: ONLY use list_itac_products
        - DO NOT call multiple tools for the same query unless explicitly needed
        - DO NOT call document_qa for weather, general questions, or non-IDC topics
        
        IMPORTANT: When calling tools, preserve the user's EXACT question including all qualifiers 
        like "detailed", "comprehensive", "full explanation", "step-by-step", etc. 
        These words are important for determining the quality and depth of the response.
        
        Do not simplify or paraphrase the user's question when calling tools.
        
        CONVERSATION CONTEXT: You can only see the current conversation history provided to you.
        If a user asks about something that happened earlier but is not visible in the current context,
        simply say "I don't know" or "I don't have that information in our current conversation."
        Do not make assumptions or ask for more information that you cannot access."""
    )
    context_messages = [system_message] + state.get("messages", [])
    
    # Ensure we have a valid prompt history to LLM
    context_messages = sanitize_message_history(context_messages)
    
    logger.info(f"🔄 [{trace_id}] Router invoked with %d messages using short-term memory.", len(context_messages))
    json_ready = [to_openai_dict(m) for m in context_messages]
    logger.info(f"📤 [{trace_id}] Final STM payload to LLM:\n%s", json.dumps(json_ready, indent=2))
    
    # Prepare callbacks for Langfuse tracing
    callbacks = []
    if langfuse_handler:
        callbacks.append(langfuse_handler)
    
    start_llm = time.time()
    ai_msg = await llm_with_tools.ainvoke(context_messages, config={"callbacks": callbacks})
    llm_duration = time.time() - start_llm
    
    if ai_msg.content and ai_msg.content.strip():
        logger.info(f"🧠 [{trace_id}] LLM responded ({llm_duration:.2f}s): %s", ai_msg.content)
    else:
        logger.info(f"🧠 [{trace_id}] LLM responded ({llm_duration:.2f}s) — no text content.")
    
    # Debug: Check if tool_calls exist
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        logger.info(f"🔧 [{trace_id}] Tool calls found in LLM response: %s", ai_msg.tool_calls)
    else:
        logger.info(f"❌ [{trace_id}] No tool_calls attribute found in LLM response")
        logger.info(f"🔍 [{trace_id}] AI message attributes: %s", dir(ai_msg))
        logger.info(f"🔍 [{trace_id}] AI message type: %s", type(ai_msg))

     # Check for low confidence
    if await is_low_confidence(ai_msg.content):
        logger.warning(f"⚠️ [{trace_id}] Low-confidence response detected — switching to long-term memory.")
        user_id = state.get("user_id")
        session_id = state.get("session_id")

        if user_id and session_id:
            ltm_history = get_last_n_messages(user_id, session_id, int(os.getenv("LONG_TERM_MEMORY")))
            ltm_history = sanitize_message_history(ltm_history)
            context_messages = [system_message] + ltm_history
  
            # Ensure we have a valid prompt history to LLM
            context_messages = sanitize_message_history(context_messages)
            logger.info(f"🔄 [{trace_id}] Router invoked with %d messages using long-term memory.", len(context_messages))
            json_ready = [to_openai_dict(m) for m in context_messages]
            logger.info(f"📤 [{trace_id}] Final LTM payload to LLM:\n%s", json.dumps(json_ready, indent=2))
            
            # Retry with long-term memory
            start_ltm = time.time()
            ai_msg = await llm_with_tools.ainvoke(context_messages, config={"callbacks": callbacks})
            ltm_duration = time.time() - start_ltm
            logger.info(f"🔁 [{trace_id}] Retried with LTM ({ltm_duration:.2f}s). New response: %s", ai_msg.content)
        else:
            logger.warning(f"⚠️ [{trace_id}] LTM fetch skipped due to missing user/session info.")

    return {
        **state,
        "messages": state["messages"] + [ai_msg]
    }

def extract_output(state: AgentState) -> AgentState:
    """
    Extracts the latest assistant message as output.
    """
    trace_id = state.get("trace_id", "unknown")
    start_time = state.get("start_time", time.time())
    total_duration = time.time() - start_time
    
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            logger.info(f"✅ [{trace_id}] E2E flow completed ({total_duration:.2f}s): {msg.content[:100]}...")
            return {
                **state,
                "output": msg.content,
                "messages": state["messages"] + [AIMessage(content=msg.content)]
            }
    
    logger.warning(f"⚠️ [{trace_id}] E2E flow completed ({total_duration:.2f}s) with no response")
    return {
        **state,
        "output": "⚠️ No response",
        "messages": state.get("messages", []) + [AIMessage(content="⚠️ No response")]
    }

def should_continue(state: AgentState) -> str:
    """
    Determine whether to continue to tools or go to extract_output based on tool calls.
    """
    trace_id = state.get("trace_id", "unknown")
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info(f"🔧 [{trace_id}] Tool calls detected, routing to tools")
        return "tools"
    else:
        logger.info(f"💬 [{trace_id}] No tool calls, routing to extract output")
        return "extract_output"

def build_graph():
    """
    Builds and compiles the LangGraph agent workflow.
    """
    builder = StateGraph(AgentState)
    builder.add_node("router", router)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("extract_output", extract_output)
    
    # Add conditional edge from router
    builder.add_conditional_edges(
        "router",
        should_continue,
        {
            "tools": "tools",
            "extract_output": "extract_output"
        }
    )
    
    builder.add_edge("tools", "extract_output")
    builder.add_edge("extract_output", END)
    builder.set_entry_point("router")
    return builder.compile()
