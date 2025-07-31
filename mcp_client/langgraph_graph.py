# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

import os
import logging
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


# Load environment variables from .env file
load_dotenv()

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict, total=False):
    """
    Defines the state structure for the agent, including messages and user/session identifiers.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    output: str
    user_id: str
    session_id: str
    
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
    
    logger.info("🔄 Router invoked with %d messages using short-term memory.", len(context_messages))
    json_ready = [to_openai_dict(m) for m in context_messages]
    logger.info("📤 Final STM payload to LLM:\n%s", json.dumps(json_ready, indent=2))
    
    ai_msg = await llm_with_tools.ainvoke(context_messages)
    if ai_msg.content and ai_msg.content.strip():
        logger.info("🧠 LLM responded: %s", ai_msg.content)
    else:
        logger.info("🧠 LLM responded — no text content.")
    
    # Debug: Check if tool_calls exist
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        logger.info("🔧 Tool calls found in LLM response: %s", ai_msg.tool_calls)
    else:
        logger.info("❌ No tool_calls attribute found in LLM response")
        logger.info("🔍 AI message attributes: %s", dir(ai_msg))
        logger.info("🔍 AI message type: %s", type(ai_msg))

     # Check for low confidence
    if await is_low_confidence(ai_msg.content):
        logger.warning("⚠️ Low-confidence response detected — switching to long-term memory.")
        user_id = state.get("user_id")
        session_id = state.get("session_id")

        if user_id and session_id:
            ltm_history = get_last_n_messages(user_id, session_id, int(os.getenv("LONG_TERM_MEMORY")))
            ltm_history = sanitize_message_history(ltm_history)
            context_messages = [system_message] + ltm_history
  
            # Ensure we have a valid prompt history to LLM
            context_messages = sanitize_message_history(context_messages)
            logger.info("🔄 Router invoked with %d messages using long-term memory.", len(context_messages))
            json_ready = [to_openai_dict(m) for m in context_messages]
            logger.info("📤 Final LTM payload to LLM:\n%s", json.dumps(json_ready, indent=2))
            
            # Retry with long-term memory
            ai_msg = await llm_with_tools.ainvoke(context_messages)
            logger.info("🔁 Retried with LTM. New response: %s", ai_msg.content)
        else:
            logger.warning("⚠️ LTM fetch skipped due to missing user/session info.")

    return {
        **state,
        "messages": state["messages"] + [ai_msg]
    }

def extract_output(state: AgentState) -> AgentState:
    """
    Extracts the latest assistant message as output.
    """
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return {
                **state,
                "output": msg.content,
                "messages": state["messages"] + [AIMessage(content=msg.content)]
            }
    return {
        **state,
        "output": "⚠️ No response",
        "messages": state.get("messages", []) + [AIMessage(content="⚠️ No response")]
    }

def should_continue(state: AgentState) -> str:
    """
    Determine whether to continue to tools or go to extract_output based on tool calls.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("🔧 Tool calls detected, routing to tools")
        return "tools"
    else:
        logger.info("💬 No tool calls, routing to extract output")
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
