"""
Task 3: Test the live API
Run this while the API server is running in another terminal
"""

import requests
import json

API_URL = "http://localhost:8000"

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 3: API Testing")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n🏥 Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.json()}")
    
    # Test 2: Model info
    print("\n📋 Getting model info...")
    response = requests.get(f"{API_URL}/model-info")
    model_info = response.json()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test 3: Transcription
    print("\n🎤 Testing transcription endpoint...")
    test_audio_path = "test_audio/test_1.wav"
    
    with open(test_audio_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/transcribe", files=files)
    
    result = response.json()
    print(f"\n✅ Transcription Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("✅ API is working!")
    print("=" * 50)
