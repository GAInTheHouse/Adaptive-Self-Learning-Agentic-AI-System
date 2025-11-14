"""
Task 1: Compare Whisper vs Wav2Vec2
Run this script to decide which model to deploy
"""
from src.model_selector import STTModelEvaluator
import os

# Download a small test audio file or use your own
# For testing, we'll create a dummy audio or point to existing one
TEST_AUDIO_FILES = [
    "test_audio_1.wav",
    "test_audio_2.wav"
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
    test_files = create_test_audio()
    
    evaluator = STTModelEvaluator()
    
    print("\n🔍 Comparing Whisper vs Wav2Vec2...\n")
    results = evaluator.compare_models(test_files)
    
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
