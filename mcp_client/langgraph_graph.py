from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.prebuilt.tool_node import ToolNode
import logging
from langgraph_tool_wrappers import build_tool_wrappers
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    output: str
    history: List[dict]

logger = logging.getLogger(__name__)

def get_llm():
    return ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
        max_tokens=128,
        temperature=0
    )

llm = get_llm()
tools = build_tool_wrappers()
llm_with_tools = llm.bind_tools(tools)

def update_history(state: AgentState, new_message: dict) -> AgentState:
    history = state.get("history", [])
    history.append(new_message)
    if len(history) > 100:
        history = history[-100:]
    return {**state, "history": history}

async def router(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        content="You are a helpful assistant. Use the tools when needed. Do not just repeat the user's question."
    )
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        user_content = state["messages"][-1].content.lower()
        if "previous conversation" in user_content:
            # Pass last 100 messages as context to the LLM
            context_messages = [system_message] + state["messages"][-100:]
            ai_msg = await llm_with_tools.ainvoke(context_messages)
            new_state = {
                **state,
                "messages": state["messages"] + [ai_msg],
                "next": "extract_output"
            }
            return new_state
    # Default: use only last 5 messages for context
    context_messages = [system_message] + state["messages"][-5:]
    ai_msg = await llm_with_tools.ainvoke(context_messages)
    new_state = {
        **state,
        "messages": state["messages"] + [ai_msg],
        "next": "extract_output"
    }
    return new_state

def extract_output(state: AgentState) -> AgentState:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return {
                **state,
                "output": msg.content,
                "messages": state["messages"] + [AIMessage(content=msg.content)]
            }
    return {
        **state,
        "output": "⚠️ No response",
        "messages": state["messages"] + [AIMessage(content="⚠️ No response")]
    }

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("extract_output", extract_output)

    builder.add_edge("router", "tools")
    builder.add_edge("tools", "extract_output")
    builder.add_edge("extract_output", END)

    builder.set_entry_point("router")
    return builder.compile()
