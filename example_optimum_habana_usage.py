#!/usr/bin/env python3
"""
Example usage of Optimum Habana LLM (similar to how you used vLLM-fork)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimum_habana_client import OptimumHabanaClient

def main():
    print("🔥 Optimum Habana LLM Example")
    print("=" * 35)
    
    # Initialize client (similar to vLLM client)
    client = OptimumHabanaClient("http://localhost:8080")
    
    # Check if server is running
    health = client.health_check()
    if health.get('status') != 'healthy':
        print("❌ Server not running or not healthy")
        print("   Start server with: make run-optimum-habana")
        return
    
    print(f"✅ Connected to Optimum Habana server")
    print(f"   Device: {health.get('device', 'unknown')}")
    
    # Example 1: Simple generation (like vLLM generate)
    print("\n🤖 Example 1: Simple Generation")
    try:
        result = client.generate(
            prompt="The benefits of renewable energy include",
            max_tokens=100,
            temperature=0.7
        )
        print(f"Response: {result['response']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Chat completion (OpenAI compatible)
    print("\n💬 Example 2: Chat Completion")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain how solar panels work"}
        ]
        result = client.chat_completion(messages, max_tokens=150)
        if 'choices' in result:
            response = result['choices'][0]['message']['content']
            print(f"Assistant: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Multiple generations
    print("\n🔄 Example 3: Multiple Generations")
    prompts = [
        "Climate change solutions:",
        "Future of technology:",
        "Benefits of AI:"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        try:
            result = client.generate(prompt, max_tokens=50, temperature=0.8)
            print(f"{i}. {prompt} {result['response']}")
        except Exception as e:
            print(f"{i}. Error: {e}")

if __name__ == "__main__":
    main()
