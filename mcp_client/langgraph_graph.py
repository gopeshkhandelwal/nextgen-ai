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
    
    
# === Intent Detection ===
LTM_KEYWORDS = ["previous", "earlier", "remind", "history", "continue", "past", "last time", "again"]

def keyword_match(text: str) -> bool:
    return any(keyword in text.lower() for keyword in LTM_KEYWORDS)

async def detect_ltm_intent(user_input: str) -> bool:
    return keyword_match(user_input)


# Initialize LLM and tools
llm = get_llm()
tools = build_tool_wrappers()
llm_with_tools = llm.bind_tools(tools)

async def router(state: AgentState) -> AgentState:
    """
    Router node for the agent:
    - Uses short-term memory by default.
    - If the user asks for 'previous conversation', fetches long-term memory from the database.
    """
    system_message = SystemMessage(
        content="You are a helpful assistant. Use the tools when needed. Do not just repeat the user's question."
    )
    context_messages = [system_message] + state.get("messages", [])

    # Check if the last message is a HumanMessage and if user requests previous conversation
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        user_input = state["messages"][-1].content.lower()
        if await detect_ltm_intent(user_input):            
            user_id = state.get("user_id")
            session_id = state.get("session_id")
            logger.info("🔁 Detected intent for long-term memory retrieval.")
         
            long_term_memory = int(os.getenv("LONG_TERM_MEMORY"))

            if not user_id or not session_id:
                logger.warning("Missing user_id or session_id — cannot fetch LTM.")
                return {
                    **state,
                    "output": "⚠️ Cannot recall past conversation without user/session context.",
                    "messages": state["messages"] + [AIMessage(content="⚠️ Missing user/session info.")]
                }

            logger.info("🔁 LTM triggered — fetching last %d messages from DB for %s / %s",
                        long_term_memory, user_id, session_id)

            db_history = get_last_n_messages(user_id, session_id, long_term_memory)

            context_messages = [system_message] + [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in db_history
            ]

    # Call the LLM with the constructed context
    ai_msg = await llm_with_tools.ainvoke(context_messages)
    new_state = {
        **state,
        "messages": state["messages"] + [ai_msg],
        "next": "extract_output"
    }
    return new_state

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