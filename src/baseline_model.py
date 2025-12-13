# src/baseline_model.py
"""
Task 2: Load and deploy baseline STT model for inference
Wraps the selected model (Whisper, Wav2Vec2) for consistent inference
Supports PyTorch framework
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineSTTModel:
    """Baseline STT inference wrapper - supports Whisper and Wav2Vec2 (PyTorch)"""
    
    def __init__(self, model_name="whisper", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.framework = "pytorch"
        self.model = None
        self.processor = None
        self.model_path = None  # Track the actual model path for verification
        self.is_ctc = False
        
        # Map model names to actual models
        model_map = {
            "whisper": "openai/whisper-base",
            "whisper-base": "openai/whisper-base",
            "whisper-tiny": "openai/whisper-tiny",
            "whisper-small": "openai/whisper-small",
            # Base model now uses wav2vec2-base-960h
            "wav2vec2-base": "facebook/wav2vec2-base-960h",
        }
        
        # Check for fine-tuned model in models folder
        # Handle both legacy names and versioned names (e.g., wav2vec2-finetuned-v1)
        finetuned_path = None
        
        if model_name == "wav2vec2-finetuned" or model_name == "Fine-tuned Wav2Vec2":
            # Legacy: try current model first, then legacy path
            from src.utils.model_versioning import get_current_model_path
            current_path = get_current_model_path()
            if current_path:
                finetuned_path = Path(current_path)
            else:
                finetuned_path = Path("models/finetuned_wav2vec2")
        elif model_name.startswith("wav2vec2-finetuned-v"):
            # Versioned model name (e.g., wav2vec2-finetuned-v1)
            version_match = re.match(r'wav2vec2-finetuned-v(\d+)', model_name)
            if version_match:
                version_num = version_match.group(1)
                finetuned_path = Path(f"models/finetuned_wav2vec2_v{version_num}")
        
        if finetuned_path and finetuned_path.exists():
            try:
                from src.agent.fine_tuner import FineTuner
                logger.info(f"Loading fine-tuned Wav2Vec2 model from {finetuned_path}")
                self.model, self.processor = FineTuner.load_model(str(finetuned_path), device=self.device)
                self.is_ctc = True
                self.model_path = str(finetuned_path)  # Track that we loaded from fine-tuned path
                
                # Load metadata to get the actual model name
                metadata_file = finetuned_path / "model_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    # Always use a user-friendly name, not HuggingFace paths
                    metadata_model_name = metadata.get("model_name", "")
                    # Only use metadata name if it's not a HuggingFace path
                    if metadata_model_name and "facebook" not in metadata_model_name.lower() and "/" not in metadata_model_name:
                        self.model_name = metadata_model_name
                    else:
                        self.model_name = "Fine-tuned Wav2Vec2"
                    logger.info(f"✓ Fine-tuned model loaded successfully: {self.model_name}")
                    logger.info(f"  Model saved at: {metadata.get('saved_at', 'unknown')}")
                    base_model_in_metadata = metadata.get("model_name", "unknown")
                    if base_model_in_metadata and base_model_in_metadata != self.model_name:
                        logger.info(f"  Base model: {base_model_in_metadata}")
                else:
                    self.model_name = "Fine-tuned Wav2Vec2"
                    logger.info("✓ Fine-tuned model loaded successfully (no metadata found)")
                
                # Verify model is actually different from baseline
                param_count = sum(p.numel() for p in self.model.parameters())
                logger.info(f"  Model parameters: {param_count:,}")
                logger.info(f"  Model path: {finetuned_path}")
                logger.info(f"  ✅ Using FINE-TUNED model (different from baseline)")
                
            except Exception as e:
                logger.error(f"❌ Failed to load fine-tuned model from {finetuned_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Don't silently fallback - raise the error so caller knows
                raise RuntimeError(
                    f"Failed to load fine-tuned model from {finetuned_path}. "
                    f"Error: {str(e)}. "
                    f"Please ensure the fine-tuned model exists and is valid."
                ) from e
        elif finetuned_path:
            logger.error(f"❌ Fine-tuned model not found at {finetuned_path}!")
            logger.error(f"   Expected path: {finetuned_path.absolute()}")
            logger.error(f"   Path exists: {finetuned_path.exists()}")
            # Raise an error instead of silently falling back
            raise FileNotFoundError(
                f"Fine-tuned model not found at {finetuned_path}. "
                f"Please ensure the fine-tuned model exists at this path, or use 'wav2vec2-base' instead."
            )
        else:
            actual_model = model_map.get(model_name, "openai/whisper-base")
            
            # Load wav2vec2 (CTC) if selected
            if "wav2vec2" in actual_model:
                logger.info(f"Loading Wav2Vec2 model: {actual_model}")
                self.processor = Wav2Vec2Processor.from_pretrained(actual_model)
                self.model = Wav2Vec2ForCTC.from_pretrained(actual_model)
                self.is_ctc = True
                self.model_name = model_name  # Keep original model_name (e.g., "wav2vec2-base")
                self.model_path = actual_model  # Track the HuggingFace model ID
            else:
                logger.info(f"Loading Whisper model: {actual_model}")
                self.processor = WhisperProcessor.from_pretrained(actual_model)
                self.model = WhisperForConditionalGeneration.from_pretrained(actual_model)
                self.is_ctc = False
                self.model_name = model_name  # Keep original model_name
                self.model_path = actual_model  # Track the HuggingFace model ID
        
        # Move model to device and optimize for inference
        self.model.to(self.device)
        self.model.eval()  # Inference mode
        
        # GPU optimizations
        if self.device.startswith("cuda"):
            # Use half precision for faster inference (if GPU supports it)
            if torch.cuda.is_available():
                try:
                    # Enable TensorFloat-32 for faster computation on Ampere+ GPUs
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info(f"✅ GPU optimizations enabled on {torch.cuda.get_device_name(0)}")
                except:
                    pass
    
    def transcribe(self, audio_path: str) -> Dict[str, str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with 'transcript' and 'model' metadata
        """
        audio, sr = librosa.load(audio_path, sr=16000)
        
        if self.is_ctc:
            # Wav2Vec2 CTC path
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)[0]
        else:
            # Whisper seq2seq path
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                input_features = inputs["input_features"].to(self.device)
                if self.device.startswith("cuda"):
                    predicted_ids = self.model.generate(
                        input_features,
                        max_new_tokens=128,
                        num_beams=5,
                        use_cache=True
                    )
                else:
                    predicted_ids = self.model.generate(
                        input_features,
                        max_new_tokens=128
                    )
            transcript = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
        
        return {
            "transcript": transcript,
            "model": self.model_name,
            "version": "baseline-v1",
            "framework": self.framework
        }
    
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        param_count = sum(p.numel() for p in self.model.parameters())
        info = {
            "name": self.model_name,
            "parameters": param_count,
            "device": self.device,
            "framework": self.framework,
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Include model path to verify which model is being used
        if self.model_path:
            info["model_path"] = self.model_path
            # Indicate if this is a fine-tuned model
            if "finetuned" in self.model_path.lower() or "fine" in self.model_path.lower():
                info["is_finetuned"] = True
            else:
                info["is_finetuned"] = False
        
        return info
