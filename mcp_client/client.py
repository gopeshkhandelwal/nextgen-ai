import asyncio
import os
import sys
from contextlib import AsyncExitStack
from typing import Optional
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, script_path: str):
        server_params = StdioServerParameters(command="python3", args=[script_path], env=os.environ.copy())
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        tools = await self.session.list_tools()
        print("✅ Connected to MCP server with tools:", [t.name for t in tools.tools])

    async def process_query(self, query: str) -> str:
        if not self.session:
            return "MCP session not initialized."
        messages = [{"role": "user", "content": query}]
        tool_response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"Tool named {tool.name}",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}}
            }
        } for tool in tool_response.tools]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        final_output = []
        if tool_calls:
            for call in tool_calls:
                tool_name = call.function.name
                tool_args = eval(call.function.arguments)
                print(f"🔧 Tool call from LLM: {tool_name}({tool_args})")
                tool_result = await self.session.call_tool(tool_name, tool_args)
                messages.append({"role": "assistant", "tool_calls": [call.model_dump()]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result.content})
            followup = client.chat.completions.create(model=MODEL, messages=messages)
            final_output.append(followup.choices[0].message.content)
        else:
            final_output.append(message.content)
        return "\n".join(final_output)

    async def chat_loop(self):
        print("🧠 LLM + MCP Tool Client")
        while True:
            query = input("\nQuery (or 'quit'): ").strip()
            if query.lower() == "quit":
                break
            response = await self.process_query(query)
            print("\n📣 Response:\n", response)

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py server.py")
        return
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
