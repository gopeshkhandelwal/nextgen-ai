#!/usr/bin/env python3
"""
Quick test of the working Optimum Habana integration
"""
import os
import sys
sys.path.append('.')

from dotenv import load_dotenv
from common_utils.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def test_optimum_habana():
    print("🧪 Testing Optimum Habana with Simple Query")
    print("=" * 50)
    
    try:
        # Test with the current configuration
        llm, is_openai = get_llm(tool_mode=False)
        print(f"✅ LLM Provider: {os.getenv('LLM_PROVIDER')}")
        
        # Simple test
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is USA currency?")
        ]
        
        print("🔄 Generating response...")
        response = llm.invoke(messages)
        print(f"✅ Response: {response.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_optimum_habana()
