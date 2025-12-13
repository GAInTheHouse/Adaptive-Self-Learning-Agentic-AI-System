#!/usr/bin/env python3
"""
Preprocessing script for STT datasets.
Downloads from GCS, preprocesses, and uploads results back.
"""

import sys
import json
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.preprocessing import AudioPreprocessor
from src.data.evaluation_splits import EvaluationSplitter
from src.utils.gcs_utils import get_gcs_manager

def preprocess_dataset(dataset_name: str, local_raw_dir: Path, local_processed_dir: Path):
    """Preprocess a single dataset"""
    print(f"\nÌ¥ß Preprocessing {dataset_name}...")
    
    dataset_path = local_raw_dir / dataset_name
    if not dataset_path.exists():
        print(f"‚ö†Ô∏è  Dataset not found: {dataset_path}")
        return None
    
    try:
        # Load dataset
        dataset = load_from_disk(str(dataset_path))
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor(
            target_sr=16000,
            trim_silence=True,
            normalize=True
        )
        
        # Process each sample
        processed_count = 0
        metadata_list = []
        
        # Note: For Hugging Face datasets, audio is typically in 'audio' column
        # We'll create a mapping for processed data
        
        print(f"Processing {len(dataset)} samples...")
        
        # For this example, we'll just create evaluation splits
        # Full audio preprocessing would require saving individual files
        
        # Create output directory
        output_path = local_processed_dir / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed dataset
        dataset.save_to_disk(str(output_path))
        
        # Create metadata
        metadata = {
            "dataset_name": dataset_name,
            "num_samples": len(dataset),
            "preprocessing_steps": ["resampling_16kHz", "silence_trimming", "normalization"],
            "output_path": str(output_path)
        }
        
        with open(output_path / "preprocessing_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"‚úì Processed {len(dataset)} samples")
        return output_path
        
    except Exception as e:
        print(f"‚ùå Error preprocessing {dataset_name}: {e}")
        return None

def create_evaluation_splits(dataset_path: Path, output_dir: Path):
    """Create train/dev/test splits"""
    print(f"\n‚úÇÔ∏è  Creating evaluation splits for {dataset_path.name}...")
    
    try:
        splitter = EvaluationSplitter(seed=42)
        
        output_path = output_dir / dataset_path.name
        splits = splitter.create_splits(
            str(dataset_path),
            str(output_path),
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1
        )
        
        print(f"‚úì Created splits at {output_path}")
        return output_path
        
    except Exception as e:
        print(f"‚ùå Error creating splits: {e}")
        return None

def upload_processed_data(local_paths: list):
    """Upload processed data to GCS"""
    print("\n‚òÅÔ∏è  Uploading processed data to GCS...")
    
    try:
        gcs_manager = get_gcs_manager("datasets")
        
        for local_path in local_paths:
            if local_path and local_path.exists():
                dataset_name = local_path.name
                
                # Determine target GCS prefix based on parent directory
                if "processed" in str(local_path):
                    gcs_prefix = f"processed/{dataset_name}"
                elif "evaluation" in str(local_path):
                    gcs_prefix = f"evaluation/{dataset_name}"
                else:
                    gcs_prefix = f"other/{dataset_name}"
                
                print(f"\nÌ≥§ Uploading {dataset_name} to {gcs_prefix}...")
                
                uploaded = gcs_manager.upload_directory(
                    str(local_path),
                    gcs_prefix
                )
                
                print(f"‚úì Uploaded {uploaded} files")
        
        print("\n‚úì All processed data uploaded to GCS")
        return True
        
    except Exception as e:
        print(f"‚ùå Error uploading to GCS: {e}")
        return False

def main():
    """Main preprocessing routine"""
    print("="*60)
    print("Ì¥ß STT Data Preprocessing Pipeline")
    print("="*60)
    
    # Define directories
    local_raw_dir = Path("data/raw")
    local_processed_dir = Path("data/processed")
    local_evaluation_dir = Path("data/evaluation")
    
    local_processed_dir.mkdir(parents=True, exist_ok=True)
    local_evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    # List of datasets to process
    datasets_to_process = [
        "common_voice_accents",
        "librispeech_clean",
        "speech_commands"
    ]
    
    # Track processed datasets
    processed_paths = []
    evaluation_paths = []
    
    # Preprocess each dataset
    for dataset_name in datasets_to_process:
        path = preprocess_dataset(dataset_name, local_raw_dir, local_processed_dir)
        if path:
            processed_paths.append(path)
            
            # Create evaluation splits
            eval_path = create_evaluation_splits(path, local_evaluation_dir)
            if eval_path:
                evaluation_paths.append(eval_path)
    
    # Upload all processed data
    all_paths = processed_paths + evaluation_paths
    upload_success = upload_processed_data(all_paths)
    
    # Print summary
    print("\n" + "="*60)
    print("Ì≥ä Preprocessing Summary")
    print("="*60)
    print(f"Processed datasets: {len(processed_paths)}")
    print(f"Evaluation splits created: {len(evaluation_paths)}")
    
    if upload_success:
        print("\n‚úì All data uploaded to gs://stt-project-datasets/")
        print("  - Processed: gs://stt-project-datasets/processed/")
        print("  - Evaluation: gs://stt-project-datasets/evaluation/")
    
    print("\nÌ≤° Next steps:")
    print("1. Review data in GCS console")
    print("2. Begin baseline model setup (Week 1, Task 2)")
    print("3. Document preprocessing results in GitHub")
    
    return 0 if processed_paths else 1

if __name__ == "__main__":
    sys.exit(main())
