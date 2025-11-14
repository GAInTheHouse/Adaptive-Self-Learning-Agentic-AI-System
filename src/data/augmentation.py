"""
Noise augmentation for creating robust training data.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import random
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseAugmentor:
    """Add background noise to audio samples"""
    
    def __init__(self, noise_samples: List[str], sr: int = 16000):
        """
        Initialize augmentor.
        
        Args:
            noise_samples: List of paths to noise audio files
            sr: Sample rate
        """
        self.noise_samples = noise_samples
        self.sr = sr
        self.noise_cache = {}
        
        logger.info(f"Initialized with {len(noise_samples)} noise samples")
    
    def _load_noise(self, noise_path: str) -> np.ndarray:
        """Load and cache noise sample"""
        if noise_path not in self.noise_cache:
            noise, _ = librosa.load(noise_path, sr=self.sr)
            self.noise_cache[noise_path] = noise
        return self.noise_cache[noise_path]
    
    def add_noise(
        self,
        audio: np.ndarray,
        snr_db: float = 10.0,
        noise_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Add noise to audio at specified SNR.
        
        Args:
            audio: Clean audio array
            snr_db: Signal-to-noise ratio in decibels
            noise_path: Specific noise file to use (random if None)
        
        Returns:
            Noisy audio array
        """
        # Select noise
        if noise_path is None:
            noise_path = random.choice(self.noise_samples)
        
        noise = self._load_noise(noise_path)
        
        # Match noise length to audio
        if len(noise) < len(audio):
            # Repeat noise if too short
            repetitions = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repetitions)[:len(audio)]
        else:
            # Random crop if too long
            start = random.randint(0, len(noise) - len(audio))
            noise = noise[start:start + len(audio)]
        
        # Calculate scaling factor for target SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (snr_linear * noise_power))
        
        # Add scaled noise
        noisy_audio = audio + scale * noise
        
        return noisy_audio
    
    def augment_file(
        self,
        input_path: str,
        output_path: str,
        snr_db: float = 10.0
    ) -> dict:
        """
        Augment single audio file.
        
        Args:
            input_path: Input audio file
            output_path: Output audio file
            snr_db: Target SNR in dB
        
        Returns:
            Metadata dictionary
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sr)
        
        # Add noise
        noisy_audio = self.add_noise(audio, snr_db)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, noisy_audio, sr)
        
        return {
            "input_path": input_path,
            "output_path": output_path,
            "snr_db": snr_db,
            "duration": len(audio) / sr
        }
