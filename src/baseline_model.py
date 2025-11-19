# src/baseline_model.py
"""
Task 2: Load and deploy baseline STT model for inference
Wraps the selected model (Whisper or Wav2Vec2) for consistent inference
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from typing import Dict, Tuple

class BaselineSTTModel:
    """Baseline STT inference wrapper"""
    
    def __init__(self, model_name="whisper", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        if model_name == "whisper":
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        else:
            raise ValueError(f"Model {model_name} not yet supported")
        
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
            "version": "baseline-v1"
        }
    
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.model_name,
            "parameters": param_count,
            "device": self.device,
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
