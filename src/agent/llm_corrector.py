"""
LLM-based Error Corrector - Gemma Integration
Uses Gemma LLM for intelligent error correction and text improvement
"""

import logging
import torch
from typing import Dict, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaLLMCorrector:
    """
    LLM-based error corrector using Google's Gemma model.
    Provides intelligent error correction and text improvement for STT transcripts.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",  # Default to Mistral 7B Instruct
        device: Optional[str] = None,
        use_quantization: bool = True
    ):
        """
        Initialize Gemma LLM corrector.
        
        Args:
            model_name: HuggingFace model name for Gemma
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_quantization: Whether to use 8-bit quantization (saves memory)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        logger.info(f"Loading LLM: {model_name} on {self.device}")
        
        try:
            can_quantize = use_quantization and torch.cuda.is_available()
            if use_quantization and not torch.cuda.is_available():
                logger.warning("Quantization requested but CUDA not available; falling back to non-quantized load.")
                can_quantize = False

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model with optional quantization
            if can_quantize:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    device_map="auto"
                )
            
            self.model.eval()  # Inference mode
            
            logger.info(f"✅ LLM loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            logger.warning("Falling back to rule-based correction only")
            self.model = None
            self.tokenizer = None
    
    def correct_transcript(
        self,
        transcript: str,
        errors: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Use Gemma LLM to intelligently correct transcript errors.
        
        Args:
            transcript: Original transcript with errors
            errors: List of detected errors (from ErrorDetector)
            context: Additional context (audio length, confidence, etc.)
        
        Returns:
            Dictionary with corrected transcript and metadata
        """
        if not self.model or not self.tokenizer:
            logger.warning("Gemma model not available, skipping LLM correction")
            return {
                "corrected_transcript": transcript,
                "correction_method": "none",
                "llm_used": False
            }
        
        try:
            # Build prompt for Gemma
            prompt = self._build_correction_prompt(transcript, errors, context)
            
            # Generate correction
            corrected_text = self._generate_correction(prompt)
            
            return {
                "corrected_transcript": corrected_text,
                "correction_method": "gemma_llm",
                "llm_used": True,
                "original_transcript": transcript,
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
        Build a prompt for Gemma to correct the transcript.
        
        Args:
            transcript: Original transcript
            errors: List of detected errors
            context: Additional context
        
        Returns:
            Formatted prompt string
        """
        error_summary = []
        for error in errors[:5]:  # Limit to top 5 errors
            error_type = error.get('type', 'unknown')
            description = error.get('description', '')
            error_summary.append(f"- {error_type}: {description}")
        
        error_list = "\n".join(error_summary) if error_summary else "No specific errors detected, but text may need improvement."
        
        prompt = f"""You are a careful, concise transcription corrector for the medical domain.

Original transcript (medical context; may contain misspellings or nonsense words):
"{transcript}"

Detected issues:
{error_list}

Requirements:
- Output exactly one corrected sentence (no lists, no explanations).
- Make the sentence fluent, grammatical English, and keep it clearly medical in meaning.
- If words look garbled, infer the most plausible intended medical terms.
- Preserve the original meaning and clinical context; keep medical terminology if possible.
- Fix capitalization and punctuation.
- Ensure at least one change from the original if the original is garbled.
- Do NOT add a prefix/suffix; return only the corrected sentence.

Corrected sentence:"""
        
        return prompt
    
    def _generate_correction(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate correction using Gemma model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
        
        Returns:
            Corrected text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        # Ensure inputs are on the same device as the model (handles mps/cuda/cpu)
        inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.2,  # more deterministic
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up the output (remove extra formatting)
        corrected_text = generated_text.strip()
        
        # Remove any prompt-like artifacts
        corrected_text = re.sub(r'^(Corrected (transcript|sentence):)\s*', '', corrected_text, flags=re.IGNORECASE)
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
        if not self.model or not self.tokenizer:
            return transcript
        
        improvement_instructions = {
            "general": "Improve readability, fix errors, and add appropriate punctuation.",
            "punctuation": "Add appropriate punctuation marks to improve readability.",
            "capitalization": "Fix capitalization to follow standard English conventions."
        }
        
        instruction = improvement_instructions.get(improvement_type, improvement_instructions["general"])
        
        prompt = f"""You are a helpful assistant that improves speech-to-text transcriptions.

Original transcript: "{transcript}"

Please improve this transcript by: {instruction}
Maintain the original meaning and content.

Improved transcript:"""
        
        try:
            return self._generate_correction(prompt, max_length=256)
        except Exception as e:
            logger.error(f"Transcript improvement failed: {e}")
            return transcript
    
    def is_available(self) -> bool:
        """Check if Gemma model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_available():
            return {
                "model": None,
                "status": "not_loaded",
                "device": self.device
            }
        
        return {
            "model": self.model_name,
            "status": "loaded",
            "device": self.device,
            "quantization": self.use_quantization,
            "parameters": "2B" if "2b" in self.model_name.lower() else "7B" if "7b" in self.model_name.lower() else "unknown"
        }

