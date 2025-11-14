#!/usr/bin/env python3
"""
Setup script for STT Agentic AI development environment.
Configures Google Cloud Storage access, verifies GPU, and installs dependencies.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Google Cloud configuration
GCP_PROJECT_ID = "stt-agentic-ai-2025"
GCS_BUCKETS = {
    "datasets": "gs://stt-project-datasets",
    "models": "gs://stt-project-models",
    "logs": "gs://stt-project-logs"
}

def check_python_version():
    """Verify Python version is 3.8+"""
    print("Ì∞ç Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("‚ùå Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"‚úì Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\nÌ≥¶ Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("‚úì Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"‚ùå Failed to install dependencies: {e}")
        return False

def verify_gpu_access():
    """Check if GPU is available"""
    print("\nÌæÆ Checking GPU access...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"‚úì GPU detected: {gpu_name}")
            print(f"‚úì GPU count: {gpu_count}")
            print(f"‚úì CUDA version: {torch.version.cuda}")
            return True
        else:
            print("‚ö†Ô∏è  No GPU detected. Running on CPU only.")
            return False
    except ImportError:
        print("‚ö†Ô∏è  PyTorch not installed yet. Install dependencies first.")
        return False

def setup_gcp_credentials():
    """Configure Google Cloud credentials"""
    print("\n‚òÅÔ∏è  Setting up Google Cloud credentials...")
    
    # Check if gcloud is installed
    try:
        result = subprocess.run(
            ["gcloud", "version"],
            capture_output=True,
            text=True,
            check=True
        )
        print("‚úì gcloud CLI detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("‚ùå gcloud CLI not found. Install from: https://cloud.google.com/sdk/install")
        return False
    
    # Set project
    try:
        subprocess.run(
            ["gcloud", "config", "set", "project", GCP_PROJECT_ID],
            check=True
        )
        print(f"‚úì GCP project set to: {GCP_PROJECT_ID}")
    except subprocess.CalledProcessError as e:
        print(f"‚ùå Failed to set GCP project: {e}")
        return False
    
    # Authenticate application default credentials
    try:
        subprocess.run(
            ["gcloud", "auth", "application-default", "login"],
            check=True
        )
        print("‚úì Application default credentials configured")
    except subprocess.CalledProcessError as e:
        print(f"‚ö†Ô∏è  Authentication may be required: {e}")
    
    return True

def verify_gcs_buckets():
    """Verify access to Google Cloud Storage buckets"""
    print("\nÌ∫£ Verifying GCS bucket access...")
    
    try:
        from google.cloud import storage
        client = storage.Client(project=GCP_PROJECT_ID)
        
        for bucket_name, bucket_path in GCS_BUCKETS.items():
            bucket_id = bucket_path.replace("gs://", "")
            try:
                bucket = client.get_bucket(bucket_id)
                print(f"‚úì Access verified: {bucket_path}")
            except Exception as e:
                print(f"‚ùå Cannot access {bucket_path}: {e}")
                return False
        
        return True
    except ImportError:
        print("‚ö†Ô∏è  google-cloud-storage not installed. Install dependencies first.")
        return False
    except Exception as e:
        print(f"‚ùå GCS verification failed: {e}")
        return False

def create_local_directories():
    """Create necessary local directories"""
    print("\nÌ≥Å Creating local directories...")
    
    dirs = [
        "data/raw",
        "data/processed",
        "data/augmented",
        "data/evaluation",
        "logs",
        "models",
        "checkpoints"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"‚úì Created {len(dirs)} directories")
    return True

def create_config_file():
    """Create configuration file with GCS paths"""
    print("\n‚öôÔ∏è  Creating configuration file...")
    
    config = {
        "gcp": {
            "project_id": GCP_PROJECT_ID,
            "buckets": GCS_BUCKETS
        },
        "data": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "augmented_data_dir": "data/augmented",
            "evaluation_data_dir": "data/evaluation"
        },
        "model": {
            "checkpoint_dir": "checkpoints",
            "model_dir": "models"
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 5e-5,
            "num_epochs": 3
        }
    }
    
    config_path = Path("configs/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"‚úì Configuration saved to {config_path}")
    return True

def print_summary():
    """Print setup summary"""
    print("\n" + "="*60)
    print("Ìæâ Environment Setup Complete!")
    print("="*60)
    print("\nÌ≥ã Next Steps:")
    print("1. Run: python scripts/download_datasets.py")
    print("2. Run: python scripts/preprocess_data.py")
    print("3. Start development!")
    print("\nÌ≥ö Resources:")
    print(f"- GCP Project: {GCP_PROJECT_ID}")
    print(f"- Dataset bucket: {GCS_BUCKETS['datasets']}")
    print(f"- Model bucket: {GCS_BUCKETS['models']}")
    print("\nÌ≤° Tip: Use 'python scripts/verify_setup.py' to check your setup anytime")

def main():
    """Main setup routine"""
    print("="*60)
    print("Ì∫Ä STT Agentic AI Environment Setup")
    print("="*60)
    
    steps = [
        ("Python version", check_python_version),
        ("Dependencies", install_dependencies),
        ("GPU access", verify_gpu_access),
        ("GCP credentials", setup_gcp_credentials),
        ("GCS buckets", verify_gcs_buckets),
        ("Local directories", create_local_directories),
        ("Configuration file", create_config_file),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"‚ùå Error in {step_name}: {e}")
            results.append((step_name, False))
    
    print("\n" + "="*60)
    print("Ì≥ä Setup Summary")
    print("="*60)
    for step_name, success in results:
        status = "‚úì" if success else "‚ùå"
        print(f"{status} {step_name}")
    
    if all(success for _, success in results):
        print_summary()
        return 0
    else:
        print("\n‚ö†Ô∏è  Some steps failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
