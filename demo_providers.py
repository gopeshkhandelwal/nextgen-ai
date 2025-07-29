#!/usr/bin/env python3
"""
Demo script showing provider switching between OpenAI and Optimum Habana
"""
import os
import sys
sys.path.append('.')

from dotenv import load_dotenv
from common_utils.config import get_llm
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

def demo_provider_switching():
    print("🚀 NextGen AI - Provider Switching Demo")
    print("=" * 50)
    
    # Test different providers
    providers = ["openai", "optimum_habana"]  # Can add "vllm" if available
    
    for provider in providers:
        print(f"\n📡 Testing Provider: {provider.upper()}")
        print("-" * 30)
        
        # Set provider
        os.environ["LLM_PROVIDER"] = provider
        
        try:
            # Initialize LLM
            llm, is_openai = get_llm(tool_mode=False)
            print(f"✅ {provider} initialized successfully")
            
            # Test simple query
            message = HumanMessage(content="What is machine learning?")
            response = llm.invoke([message])
            
            print(f"📝 Response preview:")
            print(f"   {response.content[:150]}...")
            
            # Test tool binding
            llm_with_tools, _ = get_llm(tool_mode=True)
            from mcp_client.langgraph_tool_wrappers import build_tool_wrappers
            tools = build_tool_wrappers()
            bound_llm = llm_with_tools.bind_tools(tools)
            print(f"🔧 Tool binding: {len(tools)} tools bound")
            
        except Exception as e:
            print(f"❌ {provider} failed: {e}")
            continue
    
    print(f"\n🎯 Configuration Summary:")
    print(f"   Current Provider: {os.getenv('LLM_PROVIDER')}")
    print(f"   OpenAI Model: {os.getenv('OPENAI_MODEL', 'Not set')}")
    print(f"   vLLM API: {os.getenv('VLLM_API_BASE', 'Not set')}")
    print(f"   Optimum Habana API: {os.getenv('OPTIMUM_HABANA_API_BASE', 'Not set')}")

if __name__ == "__main__":
    demo_provider_switching()
