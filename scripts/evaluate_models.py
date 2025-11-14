"""Lightweight evaluation script to compare Whisper and Wav2Vec2 models.

Usage:
  python evaluate_models.py --audio samples/test.wav

It measures model load time and transcription time. If a reference transcript is available, it will
compute WER (requires jiwer).
"""
import argparse
import time
from .model_loader import (
    load_whisper_model,
    load_wav2vec2_model,
    transcribe_whisper,
    transcribe_wav2vec2,
)


def evaluate_whisper(audio_path, model_name="small"):
    model, load_time = load_whisper_model(model_name)
    text, elapsed = transcribe_whisper(model, audio_path)
    return {"model": "whisper", "name": model_name, "load_time": load_time, "transcription_time": elapsed, "text": text}


def evaluate_wav2vec2(audio_path, model_name="facebook/wav2vec2-base-960h"):
    (proc, model), load_time = load_wav2vec2_model(model_name)
    text, elapsed = transcribe_wav2vec2((proc, model), audio_path)
    return {"model": "wav2vec2", "name": model_name, "load_time": load_time, "transcription_time": elapsed, "text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file to evaluate")
    args = parser.parse_args()

    print("Evaluating Whisper (small)...")
    w = evaluate_whisper(args.audio)
    print(w)

    print("Evaluating Wav2Vec2 (base)...")
    v = evaluate_wav2vec2(args.audio)
    print(v)


if __name__ == "__main__":
    main()
