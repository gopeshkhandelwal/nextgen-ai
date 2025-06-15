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
    """
    phrases = os.getenv("LOW_CONFIDENCE_PHRASES", "")
    low_conf_phrases = [p.strip().lower() for p in phrases.split(",") if p.strip()]
    return any(phrase in content.lower() for phrase in low_conf_phrases)

# Initialize LLM and tools
llm = get_llm()
tools = build_tool_wrappers()
llm_with_tools = llm.bind_tools(tools)

async def router(state: AgentState) -> AgentState:
    """
    Router node for the agent:
    - Uses short-term memory by default.    
    - Automatically retries with long-term memory if LLM response is low-confidence.
    """
    system_message = SystemMessage(
        content="You are a helpful assistant. Use the tools when needed. Do not just repeat the user's question."
    )
    context_messages = [system_message] + state.get("messages", [])
    logger.info("🔄 Router invoked with %d messages using short-term memory.", len(context_messages))
    
    ai_msg = await llm_with_tools.ainvoke(context_messages)
    logger.info("🧠 LLM response: %s", ai_msg.content)

     # Check for low confidence
    if await is_low_confidence(ai_msg.content):
        logger.warning("⚠️ Low-confidence response detected — switching to long-term memory.")
        user_id = state.get("user_id")
        session_id = state.get("session_id")

        if user_id and session_id:
            ltm_history = get_last_n_messages(user_id, session_id, int(os.getenv("LONG_TERM_MEMORY")))
            context_messages = [system_message] + [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in ltm_history
            ]
            context_messages.append(state["messages"][-1])
            ai_msg = await llm_with_tools.ainvoke(context_messages)
            logger.info("🔁 Retried with LTM. New response: %s", ai_msg.content)
        else:
            logger.warning("⚠️ LTM fetch skipped due to missing user/session info.")

    return {
        **state,
        "messages": state["messages"] + [ai_msg],
        "next": "extract_output"
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

def build_graph():
    """
    Builds and compiles the LangGraph agent workflow.
    """
    builder = StateGraph(AgentState)
    builder.add_node("router", router)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("extract_output", extract_output)
    builder.add_edge("router", "tools")
    builder.add_edge("tools", "extract_output")
    builder.add_edge("extract_output", END)
    builder.set_entry_point("router")
    return builder.compile()