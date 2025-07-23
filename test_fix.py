#!/usr/bin/env python3
"""
Quick test to verify the response parsing fix
"""
import sys
sys.path.append('.')

from common_utils.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

print("🧪 Testing Response Fix")
print("=" * 30)

llm, _ = get_llm(tool_mode=False)
messages = [
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello")
]

print("🔄 Testing...")
response = llm.invoke(messages)
print(f"✅ Response: '{response.content}'")
print(f"📏 Length: {len(response.content)} characters")
