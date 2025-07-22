#!/usr/bin/env python3
"""
Simple test script for DialoGPT to diagnose response issues
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_dialogpt():
    """Test DialoGPT directly"""
    print("🔥 Testing DialoGPT directly...")
    
    try:
        # Load model
        model_name = "microsoft/DialoGPT-medium"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        
        # Test simple conversation
        conversation_text = "User: What is the weather like today?\nBot:"
        print(f"Input: {conversation_text}")
        
        # Tokenize
        inputs = tokenizer.encode(conversation_text, return_tensors="pt")
        print(f"Input tokens: {inputs.shape}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Generated response: '{response}'")
        
        if not response.strip():
            print("❌ Empty response generated!")
            
            # Try with different parameters
            print("Trying with different parameters...")
            with torch.no_grad():
                outputs2 = model.generate(
                    inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    top_k=50,
                    top_p=0.95,
                )
            
            response2 = tokenizer.decode(outputs2[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"Alternative response: '{response2}'")
        else:
            print("✅ Response generated successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dialogpt()
