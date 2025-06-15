import asyncio
import os
import sys
import uuid
import logging
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Optional
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph_graph import build_graph
from langchain_core.messages import HumanMessage
from db_utils import store_message, get_last_n_messages

# Load environment variables from .env file
load_dotenv()

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Read memory configuration from environment or use defaults
SHORT_TERM_MEMORY = int(os.getenv("SHORT_TERM_MEMORY", 5))
LONG_TERM_MEMORY = int(os.getenv("LONG_TERM_MEMORY", 100))

def get_or_create_session_id() -> str:
    """
    Retrieve or generate a unique session ID for the user.
    The session ID is persisted in a local file for reuse.
    """
    session_file = Path(".session")
    if session_file.exists():
        return session_file.read_text().strip()
    session_id = str(uuid.uuid4())
    session_file.write_text(session_id)
    return session_id

# Initialize session and user identifiers
SESSION_ID = get_or_create_session_id()
USER_ID = input("👤 Enter user ID: ").strip()

class MCPClient:
    """
    Handles connection and communication with the MCP server.
    """
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, server_script: str):
        """
        Connect to the MCP server using stdio transport.
        """
        server_params = StdioServerParameters(command="python3", args=[server_script], env=os.environ.copy())
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        tools = await self.session.list_tools()
        logger.info("✅ MCP server connected with tools: %s", [t.name for t in tools.tools])
        return self.session

    async def close(self):
        """
        Clean up and close all async resources.
        """
        await self.exit_stack.aclose()

async def run_mcp_client():
    """
    Main loop for interacting with the user and the LangGraph agent.
    Handles user input, message storage, agent invocation, and logging.
    """
    if len(sys.argv) < 2:
        logger.error("Usage: python client.py <server.py>")
        return

    mcp = MCPClient()
    await mcp.connect(sys.argv[1])

    # Patch tool wrappers with the current session for tool calls
    import langgraph_tool_wrappers
    langgraph_tool_wrappers.client_session = mcp.session

    graph = build_graph()

    logger.info("Session ID: %s", SESSION_ID)
    logger.info("User ID: %s", USER_ID)
    logger.info("Short-term memory size: %d", SHORT_TERM_MEMORY)
    logger.info("Long-term memory size: %d", LONG_TERM_MEMORY)
    logger.info("🤖 LangGraph Agent ready. Type your query or 'quit' to exit.")

    history = []  # In-memory short-term message history

    while True:
        user_input = input("📝 Your Query: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            logger.info("👋 Exiting. Goodbye!")
            break

        try:
            # Store the user message in the database for long-term memory
            store_message(
                user_id=USER_ID,
                session_id=SESSION_ID,
                role="user",
                content=user_input,
                metadata={"source": "cli"},
                token_count=len(user_input.split())
            )

            # Add the user message to short-term memory
            history.append(HumanMessage(content=user_input))
            logger.info("🔄 Processing your query...")

            # Build the agent state with short-term memory and identifiers
            state = {"messages": history, "user_id": USER_ID, "session_id": SESSION_ID}
            response = await graph.ainvoke(state)
            last_message = response.get("messages")[-1]
            
            # Store the assistant's response in the database
            store_message(
                user_id=USER_ID,
                session_id=SESSION_ID,
                role="assistant",
                content=last_message.content,
                metadata={"model": "LangGraph-Agent"},
                token_count=len(last_message.content.split())
            )

            # Add the assistant's response to short-term memory and trim to configured size
            history.append(last_message)
            history[:] = history[-SHORT_TERM_MEMORY:]

            # Log both short-term and long-term memory for debugging
            log_memory_debug(history, USER_ID, SESSION_ID, LONG_TERM_MEMORY)
            
            logger.info("✅ Final Response: %s", last_message.content)

        except Exception as e:
            logger.exception("⚠️ Error processing query: %s", e)

    await mcp.close()
    
    
def log_memory_debug(history, user_id, session_id, long_term_memory):
    """
    Logs the current short-term (in-memory) and long-term (database) memory
    for the user session, up to the configured limits.
    """
    logger = logging.getLogger("mcp_langgraph")
    logger.info("🧠 Short-term memory (%d messages):", len(history))
    for i, msg in enumerate(history):
        logger.info("  %d. %s", i + 1, msg.content)

    db_history = get_last_n_messages(user_id, session_id, n=long_term_memory)
    if db_history:
        logger.info("🗂️ Long-term memory (%d messages):", len(db_history))
        for i, msg in enumerate(db_history):
            logger.info("  %d. [%s] %s", i + 1, msg["role"], msg["content"])
    else:
        logger.info("No long-term memory messages found in the database.")

if __name__ == "__main__":
    asyncio.run(run_mcp_client())