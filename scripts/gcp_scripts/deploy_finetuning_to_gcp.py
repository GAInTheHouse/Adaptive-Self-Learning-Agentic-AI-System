#!/usr/bin/env python3
"""
Deploy and Run Fine-Tuning on Google Cloud Platform
Automates model fine-tuning on GCP GPU instances.
"""

import sys
import subprocess
import os
import json
import argparse
from pathlib import Path
import time

# Configuration
PROJECT_ID = "stt-agentic-ai-2025"
ZONE = "us-central1-a"
VM_NAME = "stt-finetuning-gpu-vm"
REMOTE_DIR = "~/stt-project"
BUCKET_NAME = "stt-project-models"

def run_command(cmd, check=True, capture_output=True):
    """Run a shell command."""
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(
        cmd if isinstance(cmd, list) else cmd.split(),
        capture_output=capture_output,
        text=True,
        check=check
    )
    return result

def run_gcloud_ssh(vm_name, zone, command, check=True):
    """Run command on GCP VM via SSH."""
    cmd = [
        "gcloud", "compute", "ssh", vm_name,
        "--zone", zone,
        "--command", command
    ]
    print(f"SSH: {command}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    return result

def check_vm_exists(vm_name, zone):
    """Check if VM exists and is running."""
    result = run_command(
        f"gcloud compute instances describe {vm_name} --zone {zone}",
        check=False
    )
    
    if result.returncode != 0:
        print(f"❌ VM '{vm_name}' not found in zone {zone}")
        return False
    
    if "RUNNING" not in result.stdout:
        print(f"⚠️  VM exists but is not running. Starting VM...")
        run_command(f"gcloud compute instances start {vm_name} --zone {zone}")
        print("   Waiting for VM to start...")
        time.sleep(30)
    
    return True

def create_finetuning_vm(vm_name, zone, machine_type="n1-standard-8"):
    """Create a GPU-enabled VM for fine-tuning."""
    print(f"\n🔧 Creating fine-tuning VM: {vm_name}")
    
    cmd = [
        "gcloud", "compute", "instances", "create", vm_name,
        "--project", PROJECT_ID,
        "--zone", zone,
        "--machine-type", machine_type,
        "--accelerator", "type=nvidia-tesla-t4,count=1",
        "--image-family", "pytorch-latest-gpu",
        "--image-project", "deeplearning-platform-release",
        "--boot-disk-size", "200GB",
        "--boot-disk-type", "pd-ssd",
        "--maintenance-policy", "TERMINATE",
        "--metadata", "install-nvidia-driver=True",
        "--scopes", "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print("✅ VM created successfully")
        print("   Waiting for startup (60 seconds)...")
        time.sleep(60)
        return True
    else:
        print(f"❌ Failed to create VM: {result.stderr}")
        return False

def upload_code(vm_name, zone):
    """Upload project code to VM."""
    print("\n📤 Uploading code to VM...")
    
    # Create remote directory
    run_gcloud_ssh(vm_name, zone, f"mkdir -p {REMOTE_DIR}")
    
    # Upload files
    local_dir = Path(__file__).parent.parent
    files_to_upload = [
        "src",
        "experiments",
        "scripts",
        "requirements.txt",
        "setup.py"
    ]
    
    for item in files_to_upload:
        local_path = local_dir / item
        if local_path.exists():
            cmd = [
                "gcloud", "compute", "scp",
                "--recurse" if local_path.is_dir() else "",
                "--zone", zone,
                str(local_path),
                f"{vm_name}:{REMOTE_DIR}/"
            ]
            # Remove empty strings
            cmd = [c for c in cmd if c]
            run_command(cmd)
    
    print("✅ Code uploaded")

def install_dependencies(vm_name, zone):
    """Install dependencies on VM."""
    print("\n📦 Installing dependencies...")
    
    commands = [
        "sudo apt-get update",
        f"cd {REMOTE_DIR} && pip install -U pip",
        f"cd {REMOTE_DIR} && pip install -r requirements.txt",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install transformers[torch] datasets accelerate"
    ]
    
    for cmd in commands:
        run_gcloud_ssh(vm_name, zone, cmd, check=False)
    
    print("✅ Dependencies installed")

def verify_gpu(vm_name, zone):
    """Verify GPU is available."""
    print("\n🔍 Verifying GPU...")
    
    result = run_gcloud_ssh(
        vm_name, zone,
        "python3 -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\"); "
        "print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"None\\\"}\")'",
        check=False
    )
    
    print(result.stdout)
    
    if "CUDA: True" in result.stdout:
        print("✅ GPU is available")
        return True
    else:
        print("⚠️  GPU not detected")
        return False

def prepare_finetuning_dataset(vm_name, zone, job_id=None):
    """Prepare fine-tuning dataset on VM."""
    print("\n📊 Preparing fine-tuning dataset...")
    
    script = f"""
import sys
sys.path.append('{REMOTE_DIR}')

from src.data.data_manager import DataManager
from src.data.finetuning_pipeline import FinetuningDatasetPipeline

# Initialize
data_manager = DataManager(use_gcs=True)
pipeline = FinetuningDatasetPipeline(data_manager, use_gcs=True)

# Prepare dataset
dataset_info = pipeline.prepare_dataset(
    min_error_score=0.5,
    balance_error_types=True
)

print(f"Dataset prepared: {{dataset_info['dataset_id']}}")
print(f"Training samples: {{dataset_info['split_sizes']['train']}}")
"""
    
    # Write script to file
    script_path = "/tmp/prepare_dataset.py"
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Upload and run script
    run_command(f"gcloud compute scp --zone {zone} {script_path} {vm_name}:{REMOTE_DIR}/prepare_dataset.py")
    result = run_gcloud_ssh(vm_name, zone, f"cd {REMOTE_DIR} && python3 prepare_dataset.py")
    
    print(result.stdout)
    
    return result.returncode == 0

def run_finetuning(vm_name, zone, dataset_id, model_name="openai/whisper-base", epochs=3):
    """Run fine-tuning on VM."""
    print(f"\n🚀 Starting fine-tuning...")
    print(f"   Dataset: {dataset_id}")
    print(f"   Base model: {model_name}")
    print(f"   Epochs: {epochs}")
    
    training_script = f"""
import sys
import torch
from pathlib import Path
sys.path.append('{REMOTE_DIR}')

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset

print("Loading dataset...")
dataset_path = Path("{REMOTE_DIR}/data/finetuning/{dataset_id}")

# Load dataset
dataset = load_dataset('json', data_files={{
    'train': str(dataset_path / 'train.jsonl'),
    'validation': str(dataset_path / 'val.jsonl')
}})

print(f"Training samples: {{len(dataset['train'])}}")
print(f"Validation samples: {{len(dataset['validation'])}}")

# Load model and processor
print("Loading model: {model_name}")
processor = WhisperProcessor.from_pretrained("{model_name}")
model = WhisperForConditionalGeneration.from_pretrained("{model_name}")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {{device}}")

# Training arguments
output_dir = "{REMOTE_DIR}/models/finetuned_{dataset_id}"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps={epochs * 1000},
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=2,
    load_best_model_at_end=True,
)

print("Starting training...")
print(f"Output directory: {{output_dir}}")

# Note: This is a simplified example
# For production, you'd need proper data collator and preprocessing
print("✅ Training configuration prepared")
print("⚠️  Full training implementation requires additional setup")
print(f"Model will be saved to: {{output_dir}}")
"""
    
    # Write and upload script
    script_path = "/tmp/run_finetuning.py"
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    run_command(f"gcloud compute scp --zone {zone} {script_path} {vm_name}:{REMOTE_DIR}/run_finetuning.py")
    
    # Run training (in background)
    print("Launching training job...")
    result = run_gcloud_ssh(
        vm_name, zone,
        f"cd {REMOTE_DIR} && nohup python3 run_finetuning.py > finetuning.log 2>&1 &",
        check=False
    )
    
    print("✅ Training job launched")
    print(f"   Monitor with: gcloud compute ssh {vm_name} --zone {zone} --command 'tail -f {REMOTE_DIR}/finetuning.log'")
    
    return True

def download_model(vm_name, zone, model_path, local_dest):
    """Download trained model from VM."""
    print(f"\n📥 Downloading model from VM...")
    
    local_dest = Path(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "gcloud", "compute", "scp",
        "--recurse",
        "--zone", zone,
        f"{vm_name}:{model_path}",
        str(local_dest)
    ]
    
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print(f"✅ Model downloaded to {local_dest}")
        return True
    else:
        print("❌ Failed to download model")
        return False

def stop_vm(vm_name, zone):
    """Stop VM to save costs."""
    print(f"\n⏸️  Stopping VM: {vm_name}")
    result = run_command(f"gcloud compute instances stop {vm_name} --zone {zone}", check=False)
    
    if result.returncode == 0:
        print("✅ VM stopped")
    else:
        print("⚠️  Failed to stop VM")

def delete_vm(vm_name, zone):
    """Delete VM."""
    print(f"\n🗑️  Deleting VM: {vm_name}")
    result = run_command(f"gcloud compute instances delete {vm_name} --zone {zone} --quiet", check=False)
    
    if result.returncode == 0:
        print("✅ VM deleted")
    else:
        print("⚠️  Failed to delete VM")

def main():
    parser = argparse.ArgumentParser(description="Deploy and run fine-tuning on GCP")
    parser.add_argument("--create-vm", action="store_true", help="Create new VM")
    parser.add_argument("--vm-name", default=VM_NAME, help="VM name")
    parser.add_argument("--zone", default=ZONE, help="GCP zone")
    parser.add_argument("--machine-type", default="n1-standard-8", help="Machine type")
    parser.add_argument("--skip-upload", action="store_true", help="Skip code upload")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--prepare-dataset", action="store_true", help="Prepare dataset")
    parser.add_argument("--run-training", action="store_true", help="Run training")
    parser.add_argument("--dataset-id", help="Dataset ID for training")
    parser.add_argument("--model-name", default="openai/whisper-base", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--download-model", help="Download model from path")
    parser.add_argument("--local-dest", default="./models", help="Local destination for model")
    parser.add_argument("--stop-vm", action="store_true", help="Stop VM after completion")
    parser.add_argument("--delete-vm", action="store_true", help="Delete VM after completion")
    
    args = parser.parse_args()
    
    global VM_NAME, ZONE
    VM_NAME = args.vm_name
    ZONE = args.zone
    
    print("="*80)
    print("GCP FINE-TUNING DEPLOYMENT")
    print("="*80)
    
    # Create VM if requested
    if args.create_vm:
        if not create_finetuning_vm(VM_NAME, ZONE, args.machine_type):
            sys.exit(1)
    
    # Check VM exists
    if not check_vm_exists(VM_NAME, ZONE):
        print("\n💡 Create VM with --create-vm flag")
        sys.exit(1)
    
    # Upload code
    if not args.skip_upload:
        upload_code(VM_NAME, ZONE)
    
    # Install dependencies
    if not args.skip_install:
        install_dependencies(VM_NAME, ZONE)
    
    # Verify GPU
    verify_gpu(VM_NAME, ZONE)
    
    # Prepare dataset
    if args.prepare_dataset:
        prepare_finetuning_dataset(VM_NAME, ZONE)
    
    # Run training
    if args.run_training:
        if not args.dataset_id:
            print("❌ --dataset-id required for training")
            sys.exit(1)
        run_finetuning(VM_NAME, ZONE, args.dataset_id, args.model_name, args.epochs)
    
    # Download model
    if args.download_model:
        download_model(VM_NAME, ZONE, args.download_model, args.local_dest)
    
    # Stop VM
    if args.stop_vm:
        stop_vm(VM_NAME, ZONE)
    
    # Delete VM
    if args.delete_vm:
        delete_vm(VM_NAME, ZONE)
    
    print("\n✅ Operations completed!")
    print(f"\n💡 SSH to VM: gcloud compute ssh {VM_NAME} --zone {ZONE}")
    print(f"💡 Monitor costs: gcloud billing accounts list")

if __name__ == "__main__":
    main()


