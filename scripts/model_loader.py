"""Model loader and inference helpers for Whisper and Wav2Vec2.

Functions:
 - load_whisper_model
 - load_wav2vec2_model
 - transcribe_whisper
 - transcribe_wav2vec2

This keeps the inference code separate from the API server.
"""
import time
from typing import Tuple


def load_whisper_model(name: str = "small"):
    """Lazy-load OpenAI Whisper model. Returns (model, load_time_seconds)."""
    import whisper  

    t0 = time.time()
    model = whisper.load_model(name)
    return model, time.time() - t0


def load_wav2vec2_model(model_name: str = "facebook/wav2vec2-base-960h"):
    """Load Hugging Face Wav2Vec2 processor and model. Returns ((processor, model), load_time_seconds)."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch

    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    return (processor, model), time.time() - t0


def transcribe_whisper(model, audio_path: str, **kwargs) -> Tuple[str, float]:
    """Run transcription with a loaded Whisper model. Returns (text, elapsed_seconds)."""
    t0 = time.time()
    result = model.transcribe(audio_path, **kwargs)
    return result.get("text", ""), time.time() - t0


def transcribe_wav2vec2(processor_model, audio_path: str, sampling_rate: int = 16000) -> Tuple[str, float]:
    """Run transcription with a loaded Wav2Vec2 (HF) model. Returns (text, elapsed_seconds)."""
    import soundfile as sf
    import numpy as np
    import time
    import torch

    processor, model = processor_model
    speech, sr = sf.read(audio_path)
    if sr != sampling_rate:
        import librosa

        speech = librosa.resample(speech.astype(float), sr, sampling_rate)

    input_values = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").input_values
    if torch.cuda.is_available():
        input_values = input_values.cuda()

    t0 = time.time()
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription, time.time() - t0
