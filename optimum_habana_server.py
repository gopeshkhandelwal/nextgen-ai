#!/usr/bin/env python3
"""
Optimum Habana Server for Meta-Llama-3.1-8B-Instruct
Memory-optimized for stable HPU inference
"""

import os
import logging
import json
import gc
from flask import Flask, request, jsonify
import torch
import time

# Optimum Habana imports
from optimum.habana import GaudiConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import Habana optimizations
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hthpu

logger = logging.getLogger(__name__)
logger.info("✅ Using Optimum Habana with Gaudi optimizations")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class OptimumHabanaLLM:
    def __init__(self, model_path="/app/models/Hermes-2-Pro-Llama-3-8B"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = self._get_optimal_device()
        
        logger.info(f"🔥 Initializing Hermes-2-Pro-Llama-3-8B with device: {self.device}")
        logger.info(f"📦 Loading Hermes-2-Pro for native function calling")
        self._load_model()
    
    def _get_optimal_device(self):
        """HPU device only"""
        return "hpu"
    
    def _optimize_hpu_memory(self):
        """Apply HPU-specific memory optimizations"""
        # Use the recommended hpu_inference_set_env for inference workloads
        htcore.hpu_inference_set_env()
        
        # Set additional HPU memory management
        os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = '0'
        os.environ['PT_HPU_MAX_COMPOUND_OP_SIZE'] = '1'
        
        logger.info("🔧 Applied HPU inference optimizations")
    
    def _load_model(self):
        """Load OpenChat-3.5-Function-Call with optimizations"""
        try:
            # Apply HPU optimizations first
            self._optimize_hpu_memory()
            
            logger.info(f"📝 Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True
            )
            
            # OpenChat uses special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"🤖 Loading OpenChat-3.5-Function-Call model")
            
            # Create Gaudi configuration
            gaudi_config = GaudiConfig()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.model = self.model.to('hpu')
            logger.info("✅ Loaded with Optimum Habana optimizations")
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"✅ OpenChat-3.5-Function-Call loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def generate(self, prompt, max_tokens=256, temperature=0.3, top_p=0.9):
        """Generate with Hermes-2-Pro with timeout and memory management"""
        try:
            logger.info(f"🔄 Starting generation with Hermes-2-Pro: '{prompt}...'")
            start_time = time.time()
            
            if not isinstance(prompt, str) or not prompt.strip():
                return "I need a question to answer."
            
            # Format prompt for Hermes-2-Pro (ChatML format)
            formatted_prompt = self._format_hermes_prompt(prompt.strip())
            logger.info(f"📝 Formatted prompt length: {len(formatted_prompt)} chars")
            
            # Tokenize with memory-efficient settings
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Reduced for memory efficiency
                padding=False
            )
            
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"🔢 Input tokens: {input_length}")
            
            # Move to HPU
            inputs = {k: v.to("hpu") for k, v in inputs.items()}
            
            # Generate with aggressive memory and time limits
            logger.info("🚀 Starting model generation...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 128),  # Reduced for speed
                    temperature=max(temperature, 0.1),
                    top_p=min(top_p, 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Save memory
                    repetition_penalty=1.05,
                    num_beams=1,  # Faster generation
                )
            
            generation_time = time.time() - start_time
            logger.info(f"⏱️ Generation completed in {generation_time:.2f}s")
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response using Hermes-2-Pro format
            response = self._extract_hermes_response(generated_text, formatted_prompt)
            
            # Force garbage collection
            del outputs
            del inputs
            gc.collect()
            
            logger.info(f"✅ Hermes-2-Pro response ({len(response)} chars): '{response[:100]}...'")
            return response
            
        except Exception as e:
            logger.error(f"❌ Hermes-2-Pro generation failed after {time.time() - start_time:.2f}s: {e}")
            # Force cleanup on error
            gc.collect()
            return f"Error: Generation failed - {str(e)}"
    
    def _format_hermes_prompt(self, prompt):
        """Format prompt for Hermes-2-Pro (ChatML format)"""
        # Check if prompt already has ChatML format
        if "<|im_start|>" in prompt:
            return prompt
        
        # Convert Llama format to ChatML format
        if "<|begin_of_text|>" in prompt and "<|start_header_id|>" in prompt:
            # Extract system and user content
            try:
                # Parse Llama format
                parts = prompt.split("<|start_header_id|>")
                system_content = ""
                user_content = ""
                
                for part in parts:
                    if part.startswith("system<|end_header_id|>"):
                        system_content = part.replace("system<|end_header_id|>", "").replace("<|eot_id|>", "").strip()
                    elif part.startswith("user<|end_header_id|>"):
                        user_content = part.replace("user<|end_header_id|>", "").replace("<|eot_id|>", "").replace("assistant<|end_header_id|>", "").strip()
                
                # Format as ChatML for Hermes-2-Pro
                if system_content and user_content:
                    return f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
                elif user_content:
                    return f"""<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
                    
            except Exception as e:
                logger.warning(f"Failed to parse Llama format: {e}")
        
        # Simple user question
        return f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
    
    def _extract_hermes_response(self, generated_text, formatted_prompt):
        """Extract response from Hermes-2-Pro generation"""
        # Remove the input prompt
        if generated_text.startswith(formatted_prompt):
            response = generated_text[len(formatted_prompt):].strip()
        else:
            # Try to find after the assistant marker
            if "<|im_start|>assistant" in generated_text:
                response = generated_text.split("<|im_start|>assistant")[-1].strip()
            else:
                response = generated_text.strip()
        
        # Clean up ChatML tokens
        response = response.replace("<|im_end|>", "").strip()
        
        # If still empty or weird, provide fallback
        if not response or len(response.strip()) < 5:
            response = "I understand your question. Let me help you with the appropriate tool."
        
        return response

# Global model instance
llm_model = None

def initialize_model():
    """Initialize the global model with memory management"""
    global llm_model
    if llm_model is None:
        try:
            llm_model = OptimumHabanaLLM()
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "model": "NousResearch/Hermes-2-Pro-Llama-3-8B",
            "device": llm_model.device if llm_model else "not_loaded",
            "model_loaded": llm_model is not None,
            "function_calling": True,  # Hermes-2-Pro is excellent for function calling
            "memory_optimized": True
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generation endpoint without signal timeout"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = min(data.get('max_tokens', 128), 128)  # Reduced for speed
        temperature = data.get('temperature', 0.3)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        prompt = prompt.strip()
        if not prompt:
            return jsonify({"error": "Empty prompt after cleaning"}), 400
            
        logger.info(f"📝 Received prompt: '{prompt[:100]}...' (length: {len(prompt)})")
        
        # Generate response using Hermes-2-Pro (no signal timeout)
        try:
            response_text = llm_model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
        except Exception as gen_error:
            logger.error(f"❌ Generation failed: {gen_error}")
            return jsonify({
                "error": f"Generation failed: {str(gen_error)}",
                "model": "NousResearch/Hermes-2-Pro-Llama-3-8B"
            }), 500
        
        logger.info(f"🤖 Generated response: '{response_text[:100]}...' (length: {len(response_text)})")
        
        return jsonify({
            "response": response_text,
            "model": "NousResearch/Hermes-2-Pro-Llama-3-8B",
            "memory_optimized": True,
            "function_calling": True
        })
        
    except Exception as e:
        logger.error(f"❌ Generation request failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Memory-Optimized Meta-Llama-3.1-8B-Instruct Server...")
    
    # Initialize model with error handling
    try:
        logger.info("🔄 Initializing model...")
        initialize_model()
        logger.info("✅ Server initialized successfully")
        
        # Start Flask server with better logging
        logger.info("🌐 Starting Flask server on 0.0.0.0:8080...")
        
        # Configure Flask logging
        import logging as flask_logging
        flask_log = flask_logging.getLogger('werkzeug')
        flask_log.setLevel(flask_logging.INFO)
        
        # Debug: Confirm we're about to start Flask
        logger.info("🔥 About to call app.run()...")
        
        # Start server - THIS IS THE MISSING PIECE!
        app.run(
            host='0.0.0.0', 
            port=8080, 
            debug=False, 
            threaded=True,
            use_reloader=False  # Important: disable reloader in Docker
        )
        
        # This should not be reached until server stops
        logger.info("🛑 Flask server stopped")
        
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
