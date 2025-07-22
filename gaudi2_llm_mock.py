#!/usr/bin/env python3
"""
Mock Gaudi2 LLM for local testing
This will be replaced with the actual Gaudi2 implementation when running on Gaudi2 hardware
"""

import os
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockGaudi2LLM:
    """Mock Gaudi2 LLM for local testing - simulates responses"""
    
    def __init__(self, model_path="/app/models/Llama-2-7b-chat-hf"):
        """Initialize mock Gaudi2 LLM"""
        self.model_path = model_path
        logger.info(f"🔄 Mock Gaudi2 LLM initialized (model_path: {model_path})")
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Mock chat completion - returns a simulated response"""
        
        # Extract the last user message for context
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate a mock response based on the user message
        if not user_message:
            response = "Hello! I'm a mock Gaudi2 LLM running locally for testing purposes."
        elif "weather" in user_message.lower():
            response = f"I can help you with weather information! However, as a mock LLM, I'll need to use the weather tool to get real data. The user asked: '{user_message}'"
        elif "hello" in user_message.lower() or "hi" in user_message.lower():
            response = f"Hello! I'm a mock Gaudi2 LLM. You said: '{user_message}'. I'm simulating what the real Gaudi2 LLM would respond."
        elif "tool" in user_message.lower() or "function" in user_message.lower():
            response = f"I understand you want to use tools. As a mock LLM, I can simulate tool usage. Your request: '{user_message}'"
        else:
            response = f"I'm a mock Gaudi2 LLM running locally. You asked: '{user_message}'. In a real Gaudi2 environment, I would provide a more sophisticated response using the actual Llama model."
        
        logger.info(f"🔄 Mock Gaudi2 response generated for: {user_message[:50]}...")
        return response

# Global instance
_mock_gaudi2_llm = None

def get_gaudi2_llm():
    """Get singleton mock Gaudi2 LLM instance"""
    global _mock_gaudi2_llm
    if _mock_gaudi2_llm is None:
        _mock_gaudi2_llm = MockGaudi2LLM()
    return _mock_gaudi2_llm

def chat_completion_gaudi2(messages, max_tokens=1024, temperature=0.7, **kwargs):
    """Direct function for chat completion on mock Gaudi2"""
    llm = get_gaudi2_llm()
    return llm.chat_completion(messages, max_tokens, temperature)

if __name__ == "__main__":
    # Test the mock Gaudi2 LLM
    llm = MockGaudi2LLM()
    
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = llm.chat_completion(test_messages)
    print("Test Response:", response)
