"""
Common file handling utilities used across the codebase.
Only includes functions that are actually used to avoid unnecessary abstraction.
"""

import os
import tempfile
from typing import Optional
import librosa
import logging

logger = logging.getLogger(__name__)


def load_audio_duration(audio_path: str, sample_rate: int = 16000) -> Optional[float]:
    """
    Load audio file and return its duration in seconds.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default: 16000)
    
    Returns:
        Audio duration in seconds, or None if loading fails
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return len(audio) / sr
    except Exception as e:
        logger.warning(f"Could not load audio for duration calculation: {e}")
        return None


def save_uploaded_file(file_content: bytes, suffix: str = ".wav") -> str:
    """
    Save uploaded file content to a temporary file.
    
    Args:
        file_content: File content as bytes
        suffix: File suffix (default: ".wav")
    
    Returns:
        Path to temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_content)
        return tmp.name


def cleanup_temp_file(file_path: str) -> None:
    """
    Safely remove a temporary file.
    
    Args:
        file_path: Path to temporary file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Could not remove temporary file {file_path}: {e}")
