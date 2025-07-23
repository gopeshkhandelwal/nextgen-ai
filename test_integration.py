#!/usr/bin/env python3
"""
Test script for Optimum Habana integration with Agentic AI
"""
import os
import sys
sys.path.append('.')

from dotenv import load_dotenv
from common_utils.config import get_llm
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

def test_optimum_habana_integration():
    print("🧪 Testing Optimum Habana Integration")
    print("=" * 50)
    
    # Test LLM initialization
    print("1. Testing LLM initialization...")
    try:
        llm, is_openai = get_llm(tool_mode=True)
        print(f"✅ LLM initialized successfully")
        print(f"   Type: {'OpenAI' if is_openai else 'Non-OpenAI (Optimum Habana/vLLM)'}")
        print(f"   Provider: {os.getenv('LLM_PROVIDER', 'default')}")
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return
    
    # Test bind_tools
    print("\n2. Testing tool binding...")
    try:
        from mcp_client.langgraph_tool_wrappers import build_tool_wrappers
        tools = build_tool_wrappers()
        llm_with_tools = llm.bind_tools(tools)
        print(f"✅ Tool binding successful")
        print(f"   Tools bound: {len(tools)}")
    except Exception as e:
        print(f"❌ Tool binding failed: {e}")
        return
    
    # Test simple message
    print("\n3. Testing simple message...")
    try:
        message = HumanMessage(content="Hello! Can you tell me about artificial intelligence?")
        # Test without tool mode first
        simple_llm, _ = get_llm(tool_mode=False)
        response = simple_llm.invoke([message])
        print(f"✅ Message processing successful")
        print(f"   Response type: {type(response)}")
        print(f"   Response preview: {response.content[:100]}...")
    except Exception as e:
        print(f"❌ Message processing failed: {e}")
        return
    
    print("\n🎉 All tests passed! Integration is working correctly.")
    print(f"Current provider: {os.getenv('LLM_PROVIDER', 'default')}")
    print(f"API Base: {os.getenv('OPTIMUM_HABANA_API_BASE', 'N/A')}")

if __name__ == "__main__":
    test_optimum_habana_integration()
