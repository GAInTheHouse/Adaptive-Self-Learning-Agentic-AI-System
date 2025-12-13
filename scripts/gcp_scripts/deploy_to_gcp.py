#!/usr/bin/env python3
"""
Deploy evaluation framework and code to GCP VM.
Uploads code, datasets, and runs evaluation on GPU-enabled VM.
"""

import sys
import subprocess
import os
from pathlib import Path
import argparse

# Configuration
PROJECT_ID = "stt-agentic-ai-2025"
ZONE = "us-central1-a"
VM_NAME = "stt-gpu-vm"
REMOTE_DIR = "~/stt-project"

def run_gcloud_command(cmd, check=True):
    """Run a gcloud command."""
    full_cmd = ["gcloud", "compute", "ssh", VM_NAME, "--zone", ZONE, "--command", cmd]
    print(f"Running: {' '.join(full_cmd[:4])} ... {cmd}")
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    return result

def check_vm_exists():
    """Check if VM exists and is running."""
    result = subprocess.run(
        ["gcloud", "compute", "instances", "describe", VM_NAME, "--zone", ZONE],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ VM '{VM_NAME}' not found in zone {ZONE}")
        print("   Run 'bash scripts/setup_gcp_gpu.sh' first to create the VM")
        return False
    
    # Check if running
    if "RUNNING" not in result.stdout:
        print(f"⚠️  VM exists but is not running. Starting VM...")
        subprocess.run(["gcloud", "compute", "instances", "start", VM_NAME, "--zone", ZONE])
        print("   Waiting for VM to start (30 seconds)...")
        import time
        time.sleep(30)
    
    return True

def upload_code():
    """Upload project code to VM."""
    print("\n📤 Uploading code to VM...")
    
    # Create remote directory
    run_gcloud_command(f"mkdir -p {REMOTE_DIR}")
    
    # Use gcloud compute scp to upload
    local_dir = Path(__file__).parent.parent
    subprocess.run([
        "gcloud", "compute", "scp",
        "--recurse",
        "--zone", ZONE,
        str(local_dir / "src"),
        str(local_dir / "experiments"),
        str(local_dir / "scripts"),
        str(local_dir / "requirements.txt"),
        f"{VM_NAME}:{REMOTE_DIR}/"
    ], check=True)
    
    print("✅ Code uploaded successfully")

def install_dependencies():
    """Install Python dependencies on VM."""
    print("\n📦 Installing dependencies on VM...")
    
    run_gcloud_command(
        f"cd {REMOTE_DIR} && pip install -q -r requirements.txt",
        check=False  # May have some warnings
    )
    
    print("✅ Dependencies installed")

def verify_gpu():
    """Verify GPU is available on VM."""
    print("\n🔍 Verifying GPU access...")
    
    result = run_gcloud_command(
        "python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); "
        "print(f\"GPU count: {torch.cuda.device_count()}\" if torch.cuda.is_available() else \"\")'",
        check=False
    )
    
    print(result.stdout)
    if "CUDA available: True" in result.stdout:
        print("✅ GPU is available!")
    else:
        print("⚠️  GPU not detected. Check NVIDIA drivers.")

def run_evaluation():
    """Run evaluation framework on VM."""
    print("\n🚀 Running evaluation framework on GPU...")
    
    result = run_gcloud_command(
        f"cd {REMOTE_DIR} && python3 experiments/kavya_evaluation_framework.py",
        check=False
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

def download_results():
    """Download evaluation results from VM."""
    print("\n📥 Downloading results from VM...")
    
    local_output = Path(__file__).parent.parent / "experiments" / "evaluation_outputs"
    local_output.mkdir(parents=True, exist_ok=True)
    
    subprocess.run([
        "gcloud", "compute", "scp",
        "--recurse",
        "--zone", ZONE,
        f"{VM_NAME}:{REMOTE_DIR}/experiments/evaluation_outputs/*",
        str(local_output)
    ], check=False)  # May not exist yet
    
    print("✅ Results downloaded")

def main():
    parser = argparse.ArgumentParser(description="Deploy and run STT evaluation on GCP GPU VM")
    parser.add_argument("--skip-upload", action="store_true", help="Skip code upload")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip running evaluation")
    parser.add_argument("--vm-name", default=VM_NAME, help="VM name")
    parser.add_argument("--zone", default=ZONE, help="GCP zone")
    
    args = parser.parse_args()
    
    global VM_NAME, ZONE
    VM_NAME = args.vm_name
    ZONE = args.zone
    
    print("="*60)
    print("GCP Deployment Script")
    print("="*60)
    
    if not check_vm_exists():
        sys.exit(1)
    
    if not args.skip_upload:
        upload_code()
    
    if not args.skip_install:
        install_dependencies()
    
    verify_gpu()
    
    if not args.skip_eval:
        run_evaluation()
        download_results()
    
    print("\n✅ Deployment complete!")
    print(f"\n💡 To SSH into VM: gcloud compute ssh {VM_NAME} --zone {ZONE}")

if __name__ == "__main__":
    main()

