"""
Common API helper functions to reduce duplication in FastAPI endpoints.
"""

import time
from typing import Dict, Any
from fastapi import UploadFile, HTTPException
import librosa

from ..utils.file_utils import save_uploaded_file, cleanup_temp_file, load_audio_duration


async def handle_audio_upload(file: UploadFile, suffix: str = ".wav") -> str:
    """
    Handle audio file upload and return temporary file path.
    
    Args:
        file: Uploaded file from FastAPI
        suffix: File suffix (default: ".wav")
    
    Returns:
        Path to temporary file
    
    Raises:
        HTTPException: If file reading fails
    """
    try:
        content = await file.read()
        return save_uploaded_file(content, suffix=suffix)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")


def transcribe_with_timing(model, audio_path: str, **kwargs) -> Dict[str, Any]:
    """
    Transcribe audio and add timing information.
    
    Args:
        model: Model with transcribe method
        audio_path: Path to audio file
        **kwargs: Additional arguments to pass to transcribe method
    
    Returns:
        Transcription result with inference_time_seconds added
    """
    start_time = time.time()
    result = model.transcribe(audio_path, **kwargs)
    result["inference_time_seconds"] = time.time() - start_time
    return result


def transcribe_agent_with_timing(agent, audio_path: str, enable_auto_correction: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Transcribe with agent and add timing information.
    
    Args:
        agent: Agent with transcribe_with_agent method
        audio_path: Path to audio file
        enable_auto_correction: Whether to enable auto correction
        **kwargs: Additional arguments to pass to transcribe_with_agent method
    
    Returns:
        Transcription result with timing information
    """
    # Get audio length if not provided
    audio_length = kwargs.pop('audio_length_seconds', None)
    if audio_length is None:
        audio_length = load_audio_duration(audio_path)
    
    start_time = time.time()
    result = agent.transcribe_with_agent(
        audio_path=audio_path,
        audio_length_seconds=audio_length,
        enable_auto_correction=enable_auto_correction,
        **kwargs
    )
    
    # Add timing if not already present
    if "inference_time_seconds" not in result:
        result["inference_time_seconds"] = time.time() - start_time
    
    return result

