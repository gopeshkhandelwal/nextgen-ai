#!/usr/bin/env python3
"""
Optimum Habana Server for Llama-2-7b-chat-hf
Similar to vLLM-fork but using Optimum Habana for Gaudi2
"""

import os
import logging
import json
from flask import Flask, request, jsonify
import torch

# Import with fallback strategy
logger = logging.getLogger(__name__)

try:
    # Try Optimum Habana first
    from optimum.habana.transformers import GaudiConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Import Habana optimizations
    import habana_frameworks.torch.core as htcore
    OPTIMUM_HABANA_AVAILABLE = True
    logger.info("✅ Using Optimum Habana with Gaudi optimizations")
except ImportError as e:
    logger.warning(f"⚠️  Optimum Habana import failed: {e}")
    # Fallback to standard transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    OPTIMUM_HABANA_AVAILABLE = False
    logger.info("📦 Using standard Transformers (fallback)")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class OptimumHabanaLLM:
    def __init__(self, model_path="/app/models/Llama-2-7b-chat-hf"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = self._get_optimal_device()
        
        logger.info(f"🔥 Initializing Optimum Habana LLM with device: {self.device}")
        self._load_model()
    
    def _get_optimal_device(self):
        """Determine the best available device"""
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return "hpu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load model and tokenizer with Habana optimizations"""
        try:
            logger.info(f"📝 Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"🤖 Loading model from {self.model_path}")
            
            # Configure model loading based on device
            if self.device == "hpu":
                # HPU-specific loading
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                # Move to HPU
                self.model = model.to('hpu')
                
                # Apply Habana optimizations if available
                if OPTIMUM_HABANA_AVAILABLE:
                    try:
                        # Enable mixed precision and other optimizations
                        htcore.hpu_set_env()
                        logger.info("🔥 Applied Habana optimizations")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not apply all optimizations: {e}")
                        
            elif self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Try fallback loading
            try:
                logger.info("🔄 Attempting fallback loading...")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                logger.info("✅ Model loaded with CPU fallback")
            except Exception as e2:
                logger.error(f"❌ Fallback loading also failed: {e2}")
                raise
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.9):
        """Generate text using the loaded model"""
        try:
            logger.info(f"🔄 Generating text for prompt: '{prompt[:50]}...'")
            
            # Ensure prompt is a string and not empty
            if not isinstance(prompt, str) or not prompt.strip():
                logger.warning("Empty or invalid prompt received")
                return "I need a question to answer."
            
            # Clean the prompt
            cleaned_prompt = prompt.strip()
            
            # Tokenize the input
            inputs = self.tokenizer(
                cleaned_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            
            # Move to device
            if self.device == "hpu":
                inputs = {k: v.to("hpu") for k, v in inputs.items()}
            elif self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate with explicit parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 256),
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part (remove the input prompt)
            if generated_text.startswith(cleaned_prompt):
                response = generated_text[len(cleaned_prompt):].strip()
            else:
                response = generated_text.strip()
            
            # Ensure we have a meaningful response
            if not response or len(response.strip()) < 3:
                logger.warning(f"Generated empty or very short response: '{response}'")
                return "I apologize, but I couldn't generate a proper response to your question."
            
            logger.info(f"✅ Generated response: '{response[:100]}...'")
            return response
            
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return f"Error generating response: {str(e)}"

# Global model instance
llm_model = None

def initialize_model():
    """Initialize the global model"""
    global llm_model
    if llm_model is None:
        llm_model = OptimumHabanaLLM()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": llm_model.device if llm_model else "not_loaded",
        "model_loaded": llm_model is not None
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Generate response
        response_text = llm_model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Return in OpenAI format
        return jsonify({
            "id": "cmpl-optimum-habana",
            "object": "text_completion",
            "created": 1677652288,
            "model": "llama-2-7b-chat-hf",
            "choices": [{
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Completion request failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.json
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Generate response
        response_text = llm_model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Return in OpenAI format
        return jsonify({
            "id": "chatcmpl-optimum-habana",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama-2-7b-chat-hf",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Chat completion request failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Simple text generation endpoint"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Clean and validate the prompt
        prompt = prompt.strip()
        if not prompt:
            return jsonify({"error": "Empty prompt after cleaning"}), 400
            
        logger.info(f"📝 Received prompt: '{prompt[:100]}...' (length: {len(prompt)})")
        
        # Generate response using the model
        response_text = llm_model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        logger.info(f"🤖 Generated response: '{response_text[:100]}...' (length: {len(response_text)})")
        
        return jsonify({
            "response": response_text
        })
        
    except Exception as e:
        logger.error(f"❌ Generation request failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Optimum Habana Server...")
    
    # Initialize model
    initialize_model()
    
    # Start server
    app.run(host='0.0.0.0', port=8080, debug=False)
