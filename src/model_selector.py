"""
Task 1: Evaluate Whisper vs Wav2Vec2 for cost-effectiveness and performance
Run this first in a notebook to compare models and decide which to deploy
"""

import torch
import time
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC
)
import librosa
import numpy as np

class STTModelEvaluator:
    """Compare STT models on key metrics"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.results = {}
    
    def load_whisper_base(self):
        """Load Whisper base model (~140M parameters)"""
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        model.to(self.device)
        return processor, model
    
    def load_wav2vec2_base(self):
        """Load Wav2Vec2 base model (~95M parameters, lighter)"""
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.to(self.device)
        return processor, model
    
    def benchmark_inference(self, audio_path, processor, model, model_name):
        """Measure latency and memory for a single inference"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Warm up
        _ = self._run_inference(audio, sr, processor, model, model_name)
        
        # Measure latency
        start = time.time()
        transcript = self._run_inference(audio, sr, processor, model, model_name)
        latency = time.time() - start
        
        # Memory estimate
        param_count = sum(p.numel() for p in model.parameters())
        
        return {
            "transcript": transcript,
            "latency_seconds": latency,
            "parameters": param_count,
            "model_size_mb": param_count * 4 / (1024**2)  # approximate
        }
    
    def _run_inference(self, audio, sr, processor, model, model_name):
        """Internal inference wrapper"""
        if "whisper" in model_name.lower():
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                predicted_ids = model.generate(
                    inputs["input_features"].to(self.device),
                    max_new_tokens=128
                )
            return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs["input_values"].to(self.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return processor.batch_decode(predicted_ids)[0]
    
    def compare_models(self, test_audio_paths):
        """Run comparison and return summary"""
        print("🔍 Loading models...")
        whisper_proc, whisper_model = self.load_whisper_base()
        wav2vec_proc, wav2vec_model = self.load_wav2vec2_base()
        
        print("\n📊 Benchmarking on", len(test_audio_paths), "samples...")
        
        whisper_results = []
        wav2vec_results = []
        
        for audio_path in test_audio_paths:
            try:
                w_result = self.benchmark_inference(audio_path, whisper_proc, whisper_model, "whisper")
                whisper_results.append(w_result)
            except Exception as e:
                print(f"⚠️  Whisper failed on {audio_path}: {e}")
            
            try:
                w2_result = self.benchmark_inference(audio_path, wav2vec_proc, wav2vec_model, "wav2vec2")
                wav2vec_results.append(w2_result)
            except Exception as e:
                print(f"⚠️  Wav2Vec2 failed on {audio_path}: {e}")
        
        # Aggregate metrics
        summary = {
            "whisper": {
                "avg_latency": np.mean([r["latency_seconds"] for r in whisper_results]),
                "model_size_mb": whisper_results[0]["model_size_mb"] if whisper_results else None,
                "parameters": whisper_results[0]["parameters"] if whisper_results else None,
                "samples_processed": len(whisper_results)
            },
            "wav2vec2": {
                "avg_latency": np.mean([r["latency_seconds"] for r in wav2vec_results]),
                "model_size_mb": wav2vec_results[0]["model_size_mb"] if wav2vec_results else None,
                "parameters": wav2vec_results[0]["parameters"] if wav2vec_results else None,
                "samples_processed": len(wav2vec_results)
            }
        }
        
        return summary
