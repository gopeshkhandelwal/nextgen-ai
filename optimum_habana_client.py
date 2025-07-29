#!/usr/bin/env python3
"""
Optimum Habana Client
Client to interact with Optimum Habana server (similar to vLLM client)
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimumHabanaClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        """Simple text generation"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Generate request failed: {e}")
            raise
    
    def chat_completion(self, messages, max_tokens=512, temperature=0.7):
        """OpenAI-style chat completion"""
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    def completion(self, prompt, max_tokens=512, temperature=0.7):
        """OpenAI-style completion"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            raise

def test_client():
    """Test the client functionality"""
    print("🔥 Testing Optimum Habana Client")
    print("=" * 35)
    
    client = OptimumHabanaClient()
    
    # Health check
    print("🏥 Health Check:")
    health = client.health_check()
    print(f"   Status: {health}")
    
    if health.get('status') != 'healthy':
        print("❌ Server not healthy, exiting...")
        return
    
    # Test simple generation
    print("\n🤖 Simple Generation Test:")
    try:
        result = client.generate("The future of AI is", max_tokens=100)
        print(f"   Prompt: The future of AI is")
        print(f"   Response: {result.get('response', 'No response')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test chat completion
    print("\n💬 Chat Completion Test:")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ]
        result = client.chat_completion(messages, max_tokens=150)
        if 'choices' in result:
            response = result['choices'][0]['message']['content']
            print(f"   User: What is machine learning?")
            print(f"   Assistant: {response}")
        else:
            print(f"   Response: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test completion
    print("\n📝 Completion Test:")
    try:
        result = client.completion("Explain quantum computing:", max_tokens=100)
        if 'choices' in result:
            response = result['choices'][0]['text']
            print(f"   Prompt: Explain quantum computing:")
            print(f"   Completion: {response}")
        else:
            print(f"   Response: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Client testing completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_client()
    else:
        print("Usage: python optimum_habana_client.py test")
        print("Or import and use OptimumHabanaClient class")
