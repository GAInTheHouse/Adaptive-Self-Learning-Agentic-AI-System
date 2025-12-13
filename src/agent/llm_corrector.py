"""
LLM-based Error Corrector - Ollama Integration
Uses Ollama with Llama 2/3 models for intelligent error correction and text improvement
"""

import logging
import time
from typing import Dict, Optional, List
import re

from .ollama_llm import OllamaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaLLMCorrector:
    """
    LLM-based error corrector using Ollama with Llama 2/3 models.
    Provides intelligent error correction and text improvement for STT transcripts.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b",  # Default to Ollama Llama 3.2 3B
        ollama_base_url: str = "http://localhost:11434",
        device: Optional[str] = None,
        use_quantization: bool = False,  # Not used for Ollama, kept for compatibility
        fast_mode: bool = True,  # Not used for Ollama, kept for compatibility
        raise_on_error: bool = False  # If False, mark as unavailable instead of raising
    ):
        """
        Initialize Ollama LLM corrector.
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "llama3.1:8b", "llama2:7b")
            ollama_base_url: Ollama server URL (default: http://localhost:11434)
            device: Not used for Ollama (kept for compatibility)
            use_quantization: Not used for Ollama (kept for compatibility)
            fast_mode: Not used for Ollama (kept for compatibility)
            raise_on_error: If True, raise exceptions on initialization failure. If False, mark as unavailable.
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.device = device  # Kept for compatibility
        self.use_quantization = use_quantization  # Kept for compatibility
        self.fast_mode = fast_mode  # Kept for compatibility
        
        logger.info(f"Initializing Ollama LLM corrector with model: {model_name}")
        
        try:
            self.ollama = OllamaLLM(
                model_name=model_name,
                base_url=ollama_base_url,
                raise_on_error=raise_on_error
            )
            if self.ollama.is_available():
                logger.info(f"✅ Ollama LLM corrector initialized successfully with model: {model_name}")
            else:
                logger.warning(f"⚠️  Ollama LLM corrector initialized but unavailable (server not running or model not found)")
        except Exception as e:
            if raise_on_error:
                logger.error(f"Failed to initialize Ollama LLM: {e}")
                raise  # Fail and alert if Ollama is not available and raise_on_error=True
            else:
                logger.warning(f"Failed to initialize Ollama LLM: {e}. LLM correction will be unavailable.")
                self.ollama = None
    
    def correct_transcript(
        self,
        transcript: str,
        errors: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Use Ollama LLM to intelligently correct transcript errors.
        
        Args:
            transcript: Original transcript with errors
            errors: List of detected errors (from ErrorDetector)
            context: Additional context (audio length, confidence, etc.)
        
        Returns:
            Dictionary with corrected transcript and metadata
        """
        if not self.ollama or not self.ollama.is_available():
            logger.warning("Ollama LLM not available, skipping LLM correction")
            return {
                "corrected_transcript": transcript,
                "correction_method": "none",
                "llm_used": False
            }
        
        try:
            # Build prompt for Llama
            prompt = self._build_correction_prompt(transcript, errors, context)
            
            # Generate correction with timing
            start_time = time.time()
            corrected_text = self._generate_correction(prompt)
            inference_time = time.time() - start_time
            
            # logger.info(f"LLM inference time: {inference_time:.2f}s")
            
            # If LLM returned the same text (case-insensitive), it means transcript was already correct
            # Normalize for comparison (lowercase, strip whitespace)
            original_normalized = transcript.strip().lower()
            corrected_normalized = corrected_text.strip().lower()
            
            if original_normalized == corrected_normalized:
                logger.debug("LLM returned unchanged transcript - original was already correct")
            
            return {
                "corrected_transcript": corrected_text,
                "correction_method": "ollama_llm",
                "llm_used": True,
                "original_transcript": transcript,
                "inference_time_seconds": inference_time,
                "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt
            }
            
        except Exception as e:
            logger.error(f"LLM correction failed: {e}", exc_info=True)
            return {
                "corrected_transcript": transcript,
                "correction_method": "error_fallback",
                "llm_used": False,
                "error": str(e)
            }
    
    def _build_correction_prompt(
        self,
        transcript: str,
        errors: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        """
        Build a prompt for Llama to correct the transcript.
        
        Args:
            transcript: Original transcript
            errors: List of detected errors
            context: Additional context
        
        Returns:
            Formatted prompt string
        """
        # Build error summary
        error_summary = []
        for error in errors[:5]:  # Limit to top 5 errors
            error_type = error.get('type', 'unknown')
            description = error.get('description', '')
            error_summary.append(f"- {error_type}: {description}")
        
        has_errors = len(error_summary) > 0
        if has_errors:
            error_list = "\n".join(error_summary)
            error_instruction = "Fix the errors listed above."
        else:
            error_list = "No errors detected."
            error_instruction = "If the transcript is already correct and makes sense, return it UNCHANGED. Only correct if you notice actual errors."
        
        prompt = f"""You are a careful, concise transcription corrector.

Original transcript (short conversational snippet; may contain misspellings or nonsense words):
"{transcript}"

Detected issues:
{error_list}

Requirements:
- {error_instruction}
- If the transcript is already correct, fluent, and makes sense, return it EXACTLY AS IS without any changes.
- Only correct if there are actual errors (misspellings, garbled words, grammar issues).
- Output exactly one corrected sentence (no lists, no explanations).
- Make the sentence fluent, grammatical English with natural conversational phrasing and all identifiable words.
- If words look garbled, infer the most plausible intended words based on context.
- Do NOT add a prefix/suffix; return only the corrected sentence (or unchanged original if already correct).

Corrected sentence:"""
        
        return prompt
    
    def _generate_correction(self, prompt: str, max_length: int = 256) -> str:
        """
        Generate correction using Ollama Llama model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length (not used for Ollama, kept for compatibility)
        
        Returns:
            Corrected text
        """
        # Generate using Ollama
        generated_text = self.ollama.generate(
            prompt=prompt,
            options={
                "temperature": 0.2,  # Low temperature for more deterministic output
                "num_predict": 256,  # Max tokens to generate
            }
        )
        
        # Clean up the output (remove extra formatting)
        corrected_text = generated_text.strip()
        
        # Remove any prompt-like artifacts and prefixes
        # Remove common prefixes that LLMs might add
        corrected_text = re.sub(r'^(Corrected (transcript|sentence):)\s*', '', corrected_text, flags=re.IGNORECASE)
        corrected_text = re.sub(r'^(Here is the (improved|corrected) transcript:)\s*', '', corrected_text, flags=re.IGNORECASE)
        corrected_text = re.sub(r'^(Improved transcript:)\s*', '', corrected_text, flags=re.IGNORECASE)
        corrected_text = re.sub(r'^(Improved:)\s*', '', corrected_text, flags=re.IGNORECASE)
        corrected_text = corrected_text.strip()
        
        # Remove surrounding quotes if present
        corrected_text = re.sub(r'^["\'](.*)["\']$', r'\1', corrected_text)
        corrected_text = corrected_text.strip()
        
        # If multiple lines, keep the first meaningful line
        if "\n" in corrected_text:
            lines = [ln.strip() for ln in corrected_text.splitlines() if ln.strip()]
            if lines:
                corrected_text = lines[0]
        
        return corrected_text
    
    def improve_transcript(
        self,
        transcript: str,
        improvement_type: str = "general"
    ) -> str:
        """
        Use LLM to improve transcript quality without specific error context.
        
        Args:
            transcript: Transcript to improve
            improvement_type: Type of improvement ('general', 'punctuation', 'capitalization')
        
        Returns:
            Improved transcript
        """
        if not self.ollama or not self.ollama.is_available():
            return transcript
        
        improvement_instructions = {
            "general": "Improve readability, fix errors, and add appropriate punctuation.",
            "punctuation": "Add appropriate punctuation marks to improve readability.",
            "capitalization": "Fix capitalization to follow standard English conventions."
        }
        
        instruction = improvement_instructions.get(improvement_type, improvement_instructions["general"])
        
        prompt = f"""You are a helpful assistant that improves speech-to-text transcriptions.

Original transcript: "{transcript}"

IMPORTANT: If the transcript is already correct, well-formatted, and makes sense, return it UNCHANGED.
Only make changes if there are actual errors that need fixing (missing punctuation, capitalization issues, grammar problems).

If changes are needed, improve this transcript by: {instruction}
Maintain the original meaning and content.

Do NOT add any prefix like "Here is the improved transcript:" or "Improved transcript:".
Output ONLY the improved sentence with no explanations or labels (or return unchanged if already correct).

Improved transcript:"""
        
        try:
            start_time = time.time()
            improved = self._generate_correction(prompt, max_length=256)
            inference_time = time.time() - start_time
            logger.info(f"LLM improvement inference time: {inference_time:.2f}s")
            return improved
        except Exception as e:
            logger.error(f"Transcript improvement failed: {e}")
            return transcript
    
    def is_available(self) -> bool:
        """Check if Ollama LLM is available."""
        return self.ollama is not None and hasattr(self.ollama, 'is_available') and self.ollama.is_available()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_available():
            return {
                "model": None,
                "status": "not_loaded",
                "backend": "ollama"
            }
        
        # Extract parameter count from model name
        params = "unknown"
        if "3b" in self.model_name.lower() or "3.2" in self.model_name.lower():
            params = "3B"
        elif "8b" in self.model_name.lower() or "3.1" in self.model_name.lower():
            params = "8B"
        elif "7b" in self.model_name.lower():
            params = "7B"
        
        return {
            "model": self.model_name,
            "status": "loaded",
            "backend": "ollama",
            "parameters": params,
            "base_url": self.ollama_base_url
        }

