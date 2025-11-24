"""
Task 2: Test baseline model loading and single inference
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baseline_model import BaselineSTTModel

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 2: Baseline Model Loading")
    print("=" * 50)
    
    print("\n📦 Loading Whisper baseline model...")
    model = BaselineSTTModel(model_name="whisper")
    
    print("\n📋 Model Info:")
    info = model.get_model_info()
    for key, value in info.items():
        if isinstance(value, int) and value > 1000:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # Test with the test audio we created
    test_audio = "test_audio/addf8-Alaw-GW.wav"
    
    if os.path.exists(test_audio):
        print(f"\n🎤 Testing inference on {test_audio}...")
        result = model.transcribe(test_audio)
        
        print("\n✅ Inference Result:")
        print(f"  Transcript: {result['transcript']}")
        print(f"  Model: {result['model']}")
        print(f"  Version: {result['version']}")
    else:
        print(f"⚠️  Test audio not found at {test_audio}")
        print("   Run evaluate_models.py first to create test audio.")
    
    print("\n" + "=" * 50)
    print("✅ Baseline model ready for deployment!")
    print("=" * 50)
