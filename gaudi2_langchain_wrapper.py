#!/usr/bin/env python3
"""
LangChain-compatible wrapper for Gaudi2 LLM
Provides ChatModel interface for use with LangGraph agents
"""

import logging
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

logger = logging.getLogger(__name__)

try:
    from gaudi2_llm_simple import get_gaudi2_llm
    GAUDI2_AVAILABLE = True
    GAUDI2_TYPE = "simple"
    logger.info("🔥 Using simplified Gaudi2 LLM for real hardware")
except ImportError:
    try:
        from gaudi2_llm import get_gaudi2_llm
        GAUDI2_AVAILABLE = True
        GAUDI2_TYPE = "optimum"
    except ImportError:
        try:
            from gaudi2_llm_mock import get_gaudi2_llm
            GAUDI2_AVAILABLE = True
            GAUDI2_TYPE = "mock"
            logger.warning("⚠️  Using mock Gaudi2 LLM for local testing")
        except ImportError:
            get_gaudi2_llm = None
            GAUDI2_AVAILABLE = False
            GAUDI2_TYPE = "none"

class Gaudi2ChatWrapper(BaseChatModel):
    """LangChain-compatible wrapper for Gaudi2 LLM"""
    
    model_name: str = Field(default="gaudi2-llama2")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1024)
    gaudi2_llm: Any = Field(default=None, exclude=True)  # Exclude from Pydantic serialization
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not GAUDI2_AVAILABLE:
            raise ImportError("Gaudi2 LLM not available. Make sure gaudi2_llm.py, gaudi2_llm_simple.py, or gaudi2_llm_mock.py is accessible.")
        self.gaudi2_llm = get_gaudi2_llm()
        logger.info(f"✅ Gaudi2ChatWrapper initialized ({GAUDI2_TYPE} backend)")
    
    @property
    def _llm_type(self) -> str:
        return "gaudi2"
    
    def _convert_messages_to_gaudi2_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to Gaudi2 format"""
        gaudi2_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                # Default to user for unknown message types
                role = "user"
            
            gaudi2_messages.append({
                "role": role,
                "content": message.content
            })
        
        return gaudi2_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using Gaudi2 LLM"""
        
        # Convert messages to Gaudi2 format
        gaudi2_messages = self._convert_messages_to_gaudi2_format(messages)
        
        # Extract parameters
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            # Call Gaudi2 LLM
            response = self.gaudi2_llm.chat_completion(
                messages=gaudi2_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Convert response to LangChain format
            if isinstance(response, dict) and "choices" in response:
                # OpenAI-style response
                content = response["choices"][0]["message"]["content"]
            elif isinstance(response, str):
                # Direct string response
                content = response
            else:
                content = str(response)
            
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"❌ Gaudi2 LLM generation failed: {e}")
            # Return error message as response
            error_message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate - fallback to sync for now"""
        return self._generate(messages, stop, run_manager, **kwargs)

    def _stream(self, *args, **kwargs):
        """Streaming not implemented for Gaudi2"""
        raise NotImplementedError("Streaming not yet implemented for Gaudi2ChatWrapper")
    
    def _astream(self, *args, **kwargs):
        """Async streaming not implemented for Gaudi2"""
        raise NotImplementedError("Async streaming not yet implemented for Gaudi2ChatWrapper")
    
    def bind_tools(self, tools, **kwargs):
        """Bind tools to the model - return a copy with tools attached"""
        logger.info(f"🔧 Binding {len(tools)} tools to Gaudi2ChatWrapper")
        # For now, just return self (the tools will be handled by the agent framework)
        # In a real implementation, you might want to store the tools
        return self
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

# Export the wrapper
__all__ = ["Gaudi2ChatWrapper"]
