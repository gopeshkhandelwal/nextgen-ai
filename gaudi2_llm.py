#!/usr/bin/env python3
"""
Gaudi2 LLM Interface for Agent Framework
Replaces OpenAI API calls with local Gaudi2 inference using Optimum-Habana
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.habana.transformers import GaudiTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Gaudi2LLM:
    def __init__(self, model_path="./models/Llama-2-7b-chat-hf"):
        """Initialize Gaudi2 LLM with Optimum-Habana"""
        # Use environment variable if available, otherwise use default
        self.model_path = os.getenv("LOCAL_LLM_MODEL_PATH", model_path)
        self.tokenizer = None
        self.model = None
        self.gaudi_config = None
        self._load_model()
    
    def _load_model(self):
        """Load model with Habana optimizations"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load Gaudi configuration
            self.gaudi_config = GaudiConfig()
            
            # Load model with Habana optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            logger.info("Model loaded successfully on Gaudi2")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, prompt, max_tokens=1024, temperature=0.7):
        """Generate response using Gaudi2"""
        try:
            # Prepare input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {e}"
    
    def chat_completion(self, messages, max_tokens=1024, temperature=0.7):
        """OpenAI-compatible chat completion interface"""
        try:
            # Convert messages to prompt format
            prompt = self._format_chat_prompt(messages)
            
            # Generate response
            response = self.generate_response(prompt, max_tokens, temperature)
            
            # Return OpenAI-compatible format
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }],
                "model": "llama-2-7b-chat-gaudi2",
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split())
                }
            }
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {
                "error": f"Chat completion failed: {e}"
            }
    
    def _format_chat_prompt(self, messages):
        """Format messages into Llama-2 chat prompt"""
        prompt = "<s>[INST] "
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                prompt += f"{content} [/INST] "
            elif role == "assistant":
                prompt += f"{content} </s><s>[INST] "
        
        return prompt

# Global instance for reuse
_gaudi2_llm = None

def get_gaudi2_llm():
    """Get or create Gaudi2 LLM instance"""
    global _gaudi2_llm
    if _gaudi2_llm is None:
        _gaudi2_llm = Gaudi2LLM()
    return _gaudi2_llm

def chat_completion_gaudi2(messages, max_tokens=1024, temperature=0.7, **kwargs):
    """Direct function for chat completion on Gaudi2"""
    llm = get_gaudi2_llm()
    return llm.chat_completion(messages, max_tokens, temperature)

if __name__ == "__main__":
    # Test the Gaudi2 LLM
    llm = Gaudi2LLM()
    
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = llm.chat_completion(test_messages)
    print("Test Response:", response)
