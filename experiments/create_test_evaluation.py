"""
Create a simple test evaluation dataset with ground truth for demonstration.
"""

import sys
from pathlib import Path
from datasets import Dataset, DatasetDict
import soundfile as sf
import librosa

sys.path.append(str(Path(__file__).parent.parent))

def create_test_dataset():
    """Create a simple test dataset with audio files and ground truth."""
    
    # Ground truth transcripts (from known test files)
    test_data = [
        {
            "audio_path": "test_audio/addf8-Alaw-GW.wav",
            "reference": "add the sum to the product of these three",
            "id": "test_001"
        },
        {
            "audio_path": "test_audio/test_1.wav", 
            "reference": "you",  # Simple test audio
            "id": "test_002"
        }
    ]
    
    # Load audio files and create dataset
    audio_data = []
    references = []
    ids = []
    
    for item in test_data:
        audio_path = Path(item["audio_path"])
        if audio_path.exists():
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000)
            audio_data.append({"array": audio, "sampling_rate": sr})
            references.append(item["reference"])
            ids.append(item["id"])
    
    # Create dataset
    dataset = Dataset.from_dict({
        "audio": audio_data,
        "text": references,
        "id": ids
    })
    
    # Create splits (all test for now)
    dataset_dict = DatasetDict({
        "test": dataset
    })
    
    # Save dataset
    output_path = Path("data/evaluation/test_dataset")
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    
    print(f"✅ Created test dataset with {len(dataset)} samples at {output_path}")
    return str(output_path)

if __name__ == "__main__":
    create_test_dataset()

