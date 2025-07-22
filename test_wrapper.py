#!/usr/bin/env python3
"""
Test the Gaudi2 LangChain wrapper directly
"""

import sys
import os
sys.path.append(os.getcwd())

from gaudi2_langchain_wrapper import Gaudi2ChatWrapper
from langchain_core.messages import HumanMessage, SystemMessage

def test_gaudi2_wrapper():
    """Test the LangChain wrapper"""
    print("🧪 Testing Gaudi2 LangChain wrapper...")
    
    try:
        # Create wrapper
        wrapper = Gaudi2ChatWrapper()
        print("✅ Wrapper created successfully")
        
        # Test messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the weather like today?")
        ]
        
        print("📤 Sending messages to wrapper...")
        result = wrapper._generate(messages)
        
        print(f"📥 Result type: {type(result)}")
        print(f"📥 Generations: {len(result.generations)}")
        
        if result.generations:
            generation = result.generations[0]
            print(f"📥 Generation type: {type(generation)}")
            print(f"📥 Message type: {type(generation.message)}")
            print(f"📥 Content: '{generation.message.content}'")
            print(f"📥 Content length: {len(generation.message.content)}")
        else:
            print("❌ No generations returned!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gaudi2_wrapper()
