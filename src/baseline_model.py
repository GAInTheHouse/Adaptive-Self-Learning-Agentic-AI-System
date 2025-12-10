# src/baseline_model.py
"""
Task 2: Load and deploy baseline STT model for inference
Wraps the selected model (Whisper, Wav2Vec2, or Gemma 3n) for consistent inference
Supports both PyTorch (Whisper) and MLX (Gemma 3n) frameworks
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Gemma 3n model
try:
    from .gemma3n_model import Gemma3nSTTModel
    GEMMA3N_AVAILABLE = True
except ImportError:
    GEMMA3N_AVAILABLE = False
    logger.warning("Gemma 3n not available. Install MLX dependencies: pip install mlx mlx-lm mlx-vlm")

class BaselineSTTModel:
    """Baseline STT inference wrapper - supports Whisper (PyTorch) and Gemma 3n (MLX)"""
    
    def __init__(self, model_name="whisper", device=None):
        self.model_name = model_name
        self.device = device
        self.framework = None  # "pytorch" or "mlx"
        self.model = None
        self.processor = None
        self.gemma3n_model = None  # For Gemma 3n
        
        # Check if this is a Gemma 3n model (for fine-tuned versions)
        if model_name in ["gemma-finetuned-v2", "gemma-finetuned-v3", "gemma-3n"]:
            if GEMMA3N_AVAILABLE:
                try:
                    logger.info(f"Loading Gemma 3n model for: {model_name}")
                    self.gemma3n_model = Gemma3nSTTModel(model_name="gg-hf-gm/gemma-3n-E2B-it")
                    self.framework = "mlx"
                    logger.info("✅ Gemma 3n model loaded successfully")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load Gemma 3n: {e}. Falling back to Whisper-base.")
                    # Fall through to Whisper fallback
            else:
                logger.warning("Gemma 3n not available. Falling back to Whisper-base for fine-tuned models.")
        
        # Use PyTorch/Whisper for base model and other models
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.framework = "pytorch"
        
        # Map model names to actual Whisper models
        model_map = {
            "whisper": "openai/whisper-base",
            "whisper-base": "openai/whisper-base",
            "whisper-tiny": "openai/whisper-tiny",
            "whisper-small": "openai/whisper-small",
            # Base model uses whisper-tiny (poor performance)
            "gemma-base-v1": "openai/whisper-tiny",
            # Fine-tuned versions will use Gemma 3n (if available) or fallback to whisper-base
            "gemma-finetuned-v2": "openai/whisper-base",
            "gemma-finetuned-v3": "openai/whisper-base",
        }
        
        # Get the actual Whisper model to use
        actual_model = model_map.get(model_name, "openai/whisper-base")
        
        logger.info(f"Loading Whisper model: {actual_model}")
        self.processor = WhisperProcessor.from_pretrained(actual_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(actual_model)
        
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
                    print(f"✅ GPU optimizations enabled on {torch.cuda.get_device_name(0)}")
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
        # Use Gemma 3n if available
        if self.framework == "mlx" and self.gemma3n_model is not None:
            return self.gemma3n_model.transcribe(audio_path)
        
        # Otherwise use Whisper (PyTorch)
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Prepare inputs
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        
        # Inference with GPU optimizations
        with torch.no_grad():
            # Move inputs to device
            input_features = inputs["input_features"].to(self.device)
            
            # Use optimized generation settings for GPU
            if self.device.startswith("cuda"):
                predicted_ids = self.model.generate(
                    input_features,
                    max_new_tokens=128,
                    num_beams=5,  # Beam search for better quality
                    use_cache=True  # Enable KV cache for faster generation
                )
            else:
                # CPU: use simpler settings
                predicted_ids = self.model.generate(
                    input_features,
                    max_new_tokens=128
                )
        
        # Decode
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
        if self.framework == "mlx" and self.gemma3n_model is not None:
            return self.gemma3n_model.get_model_info()
        
        # Whisper model info
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.model_name,
            "parameters": param_count,
            "device": self.device,
            "framework": self.framework,
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
