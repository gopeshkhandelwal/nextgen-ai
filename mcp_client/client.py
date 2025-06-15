import asyncio
import os
import sys
import logging
from contextlib import AsyncExitStack
from typing import Optional
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph_graph import build_graph
from langchain_core.messages import HumanMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("mcp_langgraph")

history = []

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, server_script: str):
        server_params = StdioServerParameters(command="python3", args=[server_script], env=os.environ.copy())
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        logger.info("✅ MCP server connected with tools: %s", [t.name for t in (await self.session.list_tools()).tools])
        return self.session

    async def close(self):
        await self.exit_stack.aclose()

async def main():
    print("Starting MCP client...")
    if len(sys.argv) < 2:
        logger.error("Usage: python client.py server.py")
        return

    mcp = MCPClient()
    await mcp.connect(sys.argv[1])

    import langgraph_tool_wrappers
    langgraph_tool_wrappers.client_session = mcp.session

    graph = build_graph()

    print("\n🤖 LangGraph Agent ready. Type your query or 'quit' to exit.")
    while True:
        user_input = input("📝 Your Query: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            logger.info("👋 Exiting. Goodbye!")
            break

        try:
            history.append(HumanMessage(content=user_input))
            logger.info(f"\n🔄 Processing your query...\n")
            state = {"messages": history}
            response = await graph.ainvoke(state)
            last_message = response.get("messages")[-1]
            logger.info(f"\n✅ {last_message.content}\n")

            history.append(last_message)
            # Only trim if history exceeds 100 (let agent handle trimming for state/history in output)
            if len(history) > 100:
                history[:] = history[-100:]
            
            logger.info(f"Updated History ({len(history)} messages):")
            for i, msg in enumerate(history):
                logger.info(f"  {i + 1}: {msg.content}")

        except Exception as e:
            logger.error(f"⚠️ Error: {e}\n")

    await mcp.close()

if __name__ == "__main__":
    asyncio.run(main())
