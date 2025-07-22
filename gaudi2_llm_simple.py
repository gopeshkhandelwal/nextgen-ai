#!/usr/bin/env python3
"""
Simplified Gaudi2 LLM Interface for real hardware
Uses basic transformers with Habana device detection
"""

import os
import torch
import logging
from typing import List, Dict, Any

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGaudi2LLM:
    """Simplified Gaudi2 LLM for real hardware"""
    
    def __init__(self, model_path="./models/Llama-2-7b-chat-hf"):
        """Initialize simplified Gaudi2 LLM"""
        self.model_path = os.getenv("LOCAL_LLM_MODEL_PATH", model_path)
        self.tokenizer = None
        self.model = None
        self.device = self._detect_device()
        logger.info(f"🔥 Simple Gaudi2 LLM initialized (device: {self.device})")
        self._load_model()
    
    def _detect_device(self):
        """Detect the best available device"""
        # Check for Habana devices first
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return 'hpu'
        # Fall back to CUDA if available
        elif torch.cuda.is_available():
            return 'cuda'
        # Default to CPU
        else:
            return 'cpu'
    
    def _load_model(self):
        """Load model with simple device placement"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Check if model exists locally first
            if not os.path.exists(self.model_path):
                logger.warning(f"Model path {self.model_path} not found")
                # Try a smaller, open model for testing on Gaudi2
                logger.info("Using smaller open model for Gaudi2 testing: microsoft/DialoGPT-medium")
                model_name = "microsoft/DialoGPT-medium"
            else:
                model_name = self.model_path
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with simple device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                device_map="auto" if self.device != 'cpu' else None,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Don't raise - create a dummy model for testing
            logger.warning("Creating dummy model for testing...")
            self.tokenizer = None
            self.model = None
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate chat completion"""
        
        # If model failed to load, return a helpful message
        if self.model is None or self.tokenizer is None:
            return f"🔥 Gaudi2 LLM ready but model not loaded. Please download a model or check configuration. User message: {messages[-1].get('content', '') if messages else 'No message'}"
        
        try:
            # Format messages for Llama-2-chat or general chat
            formatted_prompt = self._format_chat_prompt(messages)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if self.device != 'cpu':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with better parameters for DialoGPT
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 100),  # Shorter responses for better quality
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Avoid repetition
                    no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"🔥 Generated response on {self.device}: {response[:50]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"🔥 Gaudi2 LLM active but generation failed: {str(e)}. User message: {messages[-1].get('content', '') if messages else 'No message'}"
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for DialoGPT or general conversational format"""
        # Check if we're using DialoGPT (based on model name)
        is_dialogpt = "DialoGPT" in str(self.model.config._name_or_path) if self.model else True
        
        if is_dialogpt:
            # DialoGPT expects simple conversational format
            formatted_messages = []
            conversation_text = ""
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    # DialoGPT doesn't explicitly handle system messages, prepend as context
                    conversation_text += f"{content}\n"
                elif role == "user":
                    conversation_text += f"User: {content}\n"
                elif role == "assistant":
                    conversation_text += f"Bot: {content}\n"
            
            # Add Bot: to prompt for response
            conversation_text += "Bot:"
            return conversation_text
        else:
            # Llama-2-chat format for other models
            formatted_messages = []
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    formatted_messages.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
                elif role == "user":
                    if formatted_messages and not formatted_messages[-1].endswith("[INST] "):
                        formatted_messages.append(f"<s>[INST] {content} [/INST]")
                    else:
                        formatted_messages[-1] += f"{content} [/INST]"
                elif role == "assistant":
                    formatted_messages.append(f" {content} </s>")
            
            # If the last message is from user, we're expecting a response
            if messages and messages[-1].get("role") == "user" and not formatted_messages[-1].endswith("[/INST]"):
                formatted_messages[-1] += " [/INST]"
            
            return "".join(formatted_messages)

# Global instance
_simple_gaudi2_llm = None

def get_gaudi2_llm():
    """Get singleton simple Gaudi2 LLM instance"""
    global _simple_gaudi2_llm
    if _simple_gaudi2_llm is None:
        _simple_gaudi2_llm = SimpleGaudi2LLM()
    return _simple_gaudi2_llm

def chat_completion_gaudi2(messages, max_tokens=1024, temperature=0.7, **kwargs):
    """Direct function for chat completion on simple Gaudi2"""
    llm = get_gaudi2_llm()
    return llm.chat_completion(messages, max_tokens, temperature)

if __name__ == "__main__":
    # Test the simple Gaudi2 LLM
    llm = SimpleGaudi2LLM()
    
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = llm.chat_completion(test_messages)
    print("Test Response:", response)
