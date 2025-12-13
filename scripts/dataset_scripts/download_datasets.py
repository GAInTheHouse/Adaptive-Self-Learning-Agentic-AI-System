#!/usr/bin/env python3
"""
Download and curate STT datasets focusing on difficult cases.
Automatically uploads to Google Cloud Storage.
"""

import sys
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.gcs_utils import get_gcs_manager

# Dataset configuration
DATASETS_CONFIG = {
    "common_voice": {
        "name": "mozilla-foundation/common_voice_16_1",
        "language": "en",
        "split": "train[:5%]+validation[:50%]",  # Sample for cost efficiency
        "focus": "accents",
        "description": "Diverse accents and speakers"
    },
    "librispeech": {
        "name": "librispeech_asr",
        "config": "clean",
        "split": "test.clean[:10%]",
        "focus": "clean_baseline",
        "description": "Clean speech for augmentation baseline"
    },
    "speech_commands": {
        "name": "speech_commands",
        "config": "v0.02",
        "split": "train[:5%]",
        "focus": "short_utterances",
        "description": "Short commands with background noise"
    }
}

def download_common_voice(output_dir: Path):
    """Download and filter Common Voice dataset for accent diversity"""
    print("\nÌ≥• Downloading Common Voice (accent-focused)...")
    
    try:
        config = DATASETS_CONFIG["common_voice"]
        dataset = load_dataset(
            config["name"],
            config["language"],
            split=config["split"],
            trust_remote_code=True
        )
        
        print(f"‚úì Downloaded {len(dataset)} samples")
        
        # Filter for quality and accent diversity
        print("Ì¥ç Filtering for quality and accent diversity...")
        
        def filter_quality(example):
            # Keep samples with good upvotes/downvotes ratio
            up_votes = example.get('up_votes', 0)
            down_votes = example.get('down_votes', 0)
            return up_votes > down_votes
        
        dataset = dataset.filter(filter_quality)
        print(f"‚úì Filtered to {len(dataset)} high-quality samples")
        
        # Save locally
        local_path = output_dir / "common_voice_accents"
        dataset.save_to_disk(str(local_path))
        print(f"‚úì Saved to {local_path}")
        
        # Create metadata
        metadata = {
            "dataset_name": "common_voice_accents",
            "source": config["name"],
            "num_samples": len(dataset),
            "focus": config["focus"],
            "description": config["description"],
            "columns": dataset.column_names
        }
        
        with open(local_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return local_path, len(dataset)
        
    except Exception as e:
        print(f"‚ùå Error downloading Common Voice: {e}")
        return None, 0

def download_librispeech(output_dir: Path):
    """Download LibriSpeech for clean baseline and noise augmentation"""
    print("\nÌ≥• Downloading LibriSpeech (clean baseline)...")
    
    try:
        config = DATASETS_CONFIG["librispeech"]
        dataset = load_dataset(
            config["name"],
            config["config"],
            split=config["split"],
            trust_remote_code=True
        )
        
        print(f"‚úì Downloaded {len(dataset)} samples")
        
        # Save locally
        local_path = output_dir / "librispeech_clean"
        dataset.save_to_disk(str(local_path))
        print(f"‚úì Saved to {local_path}")
        
        # Create metadata
        metadata = {
            "dataset_name": "librispeech_clean",
            "source": config["name"],
            "num_samples": len(dataset),
            "focus": config["focus"],
            "description": config["description"],
            "columns": dataset.column_names
        }
        
        with open(local_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return local_path, len(dataset)
        
    except Exception as e:
        print(f"‚ùå Error downloading LibriSpeech: {e}")
        return None, 0

def download_speech_commands(output_dir: Path):
    """Download Speech Commands for short utterances and noise robustness"""
    print("\nÌ≥• Downloading Speech Commands...")
    
    try:
        config = DATASETS_CONFIG["speech_commands"]
        dataset = load_dataset(
            config["name"],
            config["config"],
            split=config["split"],
            trust_remote_code=True
        )
        
        print(f"‚úì Downloaded {len(dataset)} samples")
        
        # Save locally
        local_path = output_dir / "speech_commands"
        dataset.save_to_disk(str(local_path))
        print(f"‚úì Saved to {local_path}")
        
        # Create metadata
        metadata = {
            "dataset_name": "speech_commands",
            "source": config["name"],
            "num_samples": len(dataset),
            "focus": config["focus"],
            "description": config["description"],
            "columns": dataset.column_names
        }
        
        with open(local_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return local_path, len(dataset)
        
    except Exception as e:
        print(f"‚ùå Error downloading Speech Commands: {e}")
        return None, 0

def create_domain_vocabulary():
    """Create domain-specific vocabulary lists for evaluation"""
    print("\nÌ≥ù Creating domain-specific vocabulary...")
    
    domain_vocab = {
        "medical": [
            "diagnosis", "prescription", "hypertension", "radiography",
            "electrocardiogram", "stethoscope", "pharmaceutical",
            "anesthesia", "cardiovascular", "respiratory"
        ],
        "technical": [
            "authentication", "distributed", "latency", "throughput",
            "kubernetes", "containerization", "microservices",
            "asynchronous", "scalability", "infrastructure"
        ],
        "financial": [
            "amortization", "derivative", "cryptocurrency", "portfolio",
            "dividend", "depreciation", "securities", "collateral",
            "liquidity", "investment"
        ]
    }
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_path = output_dir / "domain_vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump(domain_vocab, f, indent=2)
    
    print(f"‚úì Domain vocabulary saved to {vocab_path}")
    return vocab_path

def upload_to_gcs(local_paths: list):
    """Upload downloaded datasets to Google Cloud Storage"""
    print("\n‚òÅÔ∏è  Uploading datasets to Google Cloud Storage...")
    
    try:
        gcs_manager = get_gcs_manager("datasets")
        
        for local_path in local_paths:
            if local_path and local_path.exists():
                dataset_name = local_path.name
                print(f"\nÌ≥§ Uploading {dataset_name}...")
                
                # Upload entire dataset directory
                uploaded = gcs_manager.upload_directory(
                    str(local_path),
                    f"raw/{dataset_name}"
                )
                
                print(f"‚úì Uploaded {uploaded} files for {dataset_name}")
        
        print("\n‚úì All datasets uploaded to GCS")
        return True
        
    except Exception as e:
        print(f"‚ùå Error uploading to GCS: {e}")
        return False

def create_dataset_inventory(downloaded_datasets: dict):
    """Create comprehensive inventory of downloaded datasets"""
    print("\nÌ≥ä Creating dataset inventory...")
    
    inventory = {
        "download_date": pd.Timestamp.now().isoformat(),
        "total_datasets": len(downloaded_datasets),
        "total_samples": sum(count for _, count in downloaded_datasets.values()),
        "datasets": {}
    }
    
    for dataset_name, (path, count) in downloaded_datasets.items():
        if path:
            # Load metadata if available
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            inventory["datasets"][dataset_name] = {
                "local_path": str(path),
                "gcs_path": f"gs://stt-project-datasets/raw/{path.name}",
                "num_samples": count,
                "focus": metadata.get("focus", "unknown"),
                "description": metadata.get("description", "")
            }
    
    # Save inventory
    inventory_path = Path("data/raw/dataset_inventory.json")
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=2)
    
    print(f"‚úì Inventory saved to {inventory_path}")
    
    # Upload inventory to GCS
    try:
        gcs_manager = get_gcs_manager("datasets")
        gcs_manager.upload_file(
            str(inventory_path),
            "raw/dataset_inventory.json"
        )
        print("‚úì Inventory uploaded to GCS")
    except Exception as e:
        print(f"‚ö†Ô∏è  Could not upload inventory: {e}")
    
    return inventory

def main():
    """Main download routine"""
    print("="*60)
    print("Ì≥• STT Dataset Download and Curation")
    print("="*60)
    
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    downloaded = {}
    
    # Common Voice (accents)
    path, count = download_common_voice(output_dir)
    if path:
        downloaded["common_voice"] = (path, count)
    
    # LibriSpeech (clean baseline)
    path, count = download_librispeech(output_dir)
    if path:
        downloaded["librispeech"] = (path, count)
    
    # Speech Commands (short utterances)
    path, count = download_speech_commands(output_dir)
    if path:
        downloaded["speech_commands"] = (path, count)
    
    # Create domain vocabulary
    vocab_path = create_domain_vocabulary()
    if vocab_path:
        downloaded["domain_vocab"] = (vocab_path.parent, 1)
    
    # Create inventory
    inventory = create_dataset_inventory(downloaded)
    
    # Upload to GCS
    local_paths = [path for path, _ in downloaded.values()]
    upload_success = upload_to_gcs(local_paths)
    
    # Print summary
    print("\n" + "="*60)
    print("Ì≥ä Download Summary")
    print("="*60)
    print(f"Total datasets: {len(downloaded)}")
    print(f"Total samples: {sum(count for _, count in downloaded.values())}")
    print("\nDatasets downloaded:")
    for name, (path, count) in downloaded.items():
        print(f"  ‚úì {name}: {count} samples at {path}")
    
    if upload_success:
        print("\n‚úì All datasets uploaded to gs://stt-project-datasets/raw/")
    
    print("\nÌ≤° Next step: Run 'python scripts/preprocess_data.py'")
    
    return 0 if downloaded else 1

if __name__ == "__main__":
    sys.exit(main())
