"""
Task 1: Compare Whisper vs Wav2Vec2
Run this script to decide which model to deploy
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_selector import STTModelEvaluator
import os

# Download a small test audio file or use your own
# For testing, we'll create a dummy audio or point to existing one
TEST_AUDIO_FILES = [
    "test_audio/addf8-Alaw-GW.wav",
    # "test_audio_2.wav"
]

# If you don't have test files, create a synthetic one:
def create_test_audio():
    """Generate a simple test audio file"""
    import numpy as np
    from scipy.io import wavfile
    
    sr = 16000
    duration = 3  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simple sine wave for testing
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    os.makedirs("test_audio", exist_ok=True)
    wavfile.write("test_audio/test_1.wav", sr, (audio * 32767).astype(np.int16))
    print("✅ Created test_audio/test_1.wav")
    return ["test_audio/test_1.wav"]

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 1: Model Evaluation")
    print("=" * 50)
    
    # Create test audio if needed
    # test_files = create_test_audio()
    test_files = TEST_AUDIO_FILES
    
    evaluator = STTModelEvaluator()
    
    print("\n🔍 Comparing Whisper vs Wav2Vec2...\n")
    results = evaluator.compare_models(test_files)

    print("\n🔎 SAMPLE TRANSCRIPTS:")
    print("-" * 50)

    test_file = test_files[0]
    print(f"\nAudio file: {test_file}")

    # Run both models on that audio, print their transcripts
    whisper_proc, whisper_model = evaluator.load_whisper_base()
    wav2vec_proc, wav2vec_model = evaluator.load_wav2vec2_base()

    try:
        whisper_out = evaluator.benchmark_inference(test_file, whisper_proc, whisper_model, "whisper")
        print("\nWHISPER TRANSCRIPT:")
        print(whisper_out['transcript'])
    except Exception as e:
        print(f"Whisper failed: {e}")

    try:
        wav2vec_out = evaluator.benchmark_inference(test_file, wav2vec_proc, wav2vec_model, "wav2vec2")
        print("\nWAV2VEC2 TRANSCRIPT:")
        print(wav2vec_out['transcript'])
    except Exception as e:
        print(f"Wav2Vec2 failed: {e}")

    print("-" * 50)
    
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("✅ Evaluation complete. Choose model for Task 2.")
    print("=" * 50)
