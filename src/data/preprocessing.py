"""
Audio preprocessing pipeline for STT datasets.
Handles resampling, normalization, and silence trimming.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Preprocessing pipeline for audio data"""
    
    def __init__(
        self,
        target_sr: int = 16000,
        trim_silence: bool = True,
        normalize: bool = True,
        top_db: int = 20
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_sr: Target sample rate for resampling
            trim_silence: Whether to trim leading/trailing silence
            normalize: Whether to normalize audio volume
            top_db: Threshold in dB for silence trimming
        """
        self.target_sr = target_sr
        self.trim_silence = trim_silence
        self.normalize = normalize
        self.top_db = top_db
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with librosa.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (audio array, sample rate)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            raise
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply preprocessing steps to audio.
        
        Args:
            audio: Audio array
            sr: Sample rate
        
        Returns:
            Tuple of (processed audio, metadata dict)
        """
        metadata = {
            "original_duration": len(audio) / sr,
            "original_sample_rate": sr
        }
        
        # Trim silence
        if self.trim_silence:
            audio, trimmed_indices = librosa.effects.trim(
                audio,
                top_db=self.top_db
            )
            metadata["trimmed_samples"] = len(trimmed_indices[0])
        
        # Normalize
        if self.normalize:
            audio = librosa.util.normalize(audio)
            metadata["normalized"] = True
        
        metadata["final_duration"] = len(audio) / sr
        metadata["final_sample_rate"] = sr
        
        return audio, metadata
    
    def process_file(
        self,
        input_path: str,
        output_path: str
    ) -> dict:
        """
        Process single audio file.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
        
        Returns:
            Metadata dictionary
        """
        # Load audio
        audio, sr = self.load_audio(input_path)
        
        # Preprocess
        processed_audio, metadata = self.preprocess_audio(audio, sr)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed_audio, sr)
        
        metadata["input_path"] = input_path
        metadata["output_path"] = output_path
        
        return metadata
