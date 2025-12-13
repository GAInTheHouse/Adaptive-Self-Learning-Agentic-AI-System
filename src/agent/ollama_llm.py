"""
Ollama LLM Integration
Uses Ollama to run Llama 2/3 models locally for fast inference
"""

import logging
import time
from typing import Dict, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from Ollama/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Check if Ollama is available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.error("Ollama package not found. Install with: pip install ollama")


class OllamaLLM:
    """
    Wrapper for Ollama LLM integration.
    Supports Llama 2/3 models via Ollama for fast local inference.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "llama3.1:8b", "llama2:7b")
            base_url: Ollama server URL (default: http://localhost:11434)
        
        Raises:
            ImportError: If Ollama package is not installed
            ConnectionError: If Ollama server is not running
            ValueError: If specified model is not available
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package not found. Install with: pip install ollama\n"
                "Then install Ollama: https://ollama.ai/download"
            )
        
        self.model_name = model_name
        self.base_url = base_url
        self.client = None
        
        logger.info(f"Initializing Ollama LLM with model: {model_name}")
        
        # Check Ollama server connection
        try:
            # Try to connect to Ollama server
            ollama.list()  # This will fail if server is not running
            logger.info("✓ Ollama server connection successful")
        except Exception as e:
            raise ConnectionError(
                f"Ollama server is not running or not accessible at {base_url}.\n"
                f"Error: {e}\n"
                f"Please start Ollama server with: ollama serve\n"
                f"Or install Ollama from: https://ollama.ai/download"
            )
        
        # Check if model is available
        try:
            models_response = ollama.list()
            
            # Handle Ollama's ListResponse object
            # The Ollama Python client returns a ListResponse object with a 'models' attribute
            # Each model is a Model object with a 'model' attribute (not 'name')
            # Example: ListResponse(models=[Model(model='llama3.2:3b', ...)])
            if hasattr(models_response, 'models'):
                # ListResponse object - access .models attribute
                models_list = models_response.models
            elif isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            elif isinstance(models_response, list):
                models_list = models_response
            else:
                models_list = []
            
            # Extract model names - handle Model objects, dicts, and strings
            available_models = []
            for m in models_list:
                model_name_value = None
                
                # Handle Model objects (from ollama._types.ListResponse)
                # These have a 'model' attribute (not 'name')
                if hasattr(m, 'model'):
                    model_name_value = getattr(m, 'model', None)
                elif hasattr(m, 'name'):
                    model_name_value = getattr(m, 'name', None)
                # Handle dicts
                elif isinstance(m, dict):
                    model_name_value = m.get('model') or m.get('name') or m.get('model_name')
                # Handle strings
                elif isinstance(m, str):
                    model_name_value = m
                
                if model_name_value:
                    available_models.append(model_name_value)
                    # Also add base name without tag for matching (e.g., "llama3.2" from "llama3.2:3b")
                    if ':' in model_name_value:
                        base_name = model_name_value.split(':')[0]
                        if base_name not in available_models:
                            available_models.append(base_name)
            
            # Check if model is available (try exact match first, then base name match)
            model_base_name = model_name.split(':')[0] if ':' in model_name else model_name
            model_found = False
            matched_model = None
            
            for avail_model in available_models:
                # Exact match
                if avail_model == model_name:
                    model_found = True
                    matched_model = avail_model
                    break
                # Base name match (e.g., "llama3.2" matches "llama3.2:3b")
                if avail_model == model_base_name:
                    model_found = True
                    matched_model = avail_model
                    break
                # Check if available model starts with our model name (for tags like :latest)
                if avail_model.startswith(model_base_name + ':'):
                    model_found = True
                    matched_model = avail_model
                    break
            
            if not model_found:
                raise ValueError(
                    f"Model '{model_name}' is not available in Ollama.\n"
                    f"Available models: {', '.join(available_models) if available_models else 'None (no models installed)'}\n"
                    f"Please pull the model with: ollama pull {model_name}\n"
                    f"Supported models: llama3.2:3b, llama3.1:8b, llama2:7b\n"
                    f"Or use one of the available models above."
                )
            
            logger.info(f"✓ Model '{model_name}' is available (matched: {matched_model})")
        except ValueError as e:
            # Re-raise ValueError as-is (it has helpful messages)
            raise
        except KeyError as e:
            raise RuntimeError(
                f"Failed to parse Ollama model list: unexpected structure (key: {e})\n"
                f"Please ensure Ollama is properly installed and running.\n"
                f"You can verify by running: ollama list"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to check model availability: {e}\n"
                f"Please ensure Ollama is properly installed and running.\n"
                f"You can verify by running: ollama list"
            )
        
        logger.info(f"✅ Ollama LLM initialized successfully with model: {model_name}")
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )
            
            # Validate response type
            if not isinstance(response, dict):
                raise RuntimeError(f"Unexpected response type: {type(response)}. Expected dict.")
            
            # Extract and validate result
            result = response.get('response', '')
            if not result:
                logger.warning("Ollama returned empty response")
            
            return result
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Failed to generate text with Ollama: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat completion using Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            # Validate response type
            if not isinstance(response, dict):
                raise RuntimeError(f"Unexpected response type: {type(response)}. Expected dict.")
            
            # Extract and validate result
            message = response.get('message', {})
            if not isinstance(message, dict):
                raise RuntimeError(f"Unexpected message type: {type(message)}. Expected dict.")
            
            result = message.get('content', '')
            if not result:
                logger.warning("Ollama returned empty chat response")
            
            return result
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise RuntimeError(f"Failed to chat with Ollama: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and working.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            ollama.list()
            return True
        except Exception:
            return False

