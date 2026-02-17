#!/bin/bash
# Setup script for creating a GPU-enabled GCP VM for STT model training and evaluation.
# Run this script to create a VM with GPU support.

set -e

# Configuration
PROJECT_ID="atomic-oven-478617-u7"
ZONE="us-central1-a"  # Change to your preferred zone
VM_NAME="stt-gpu-vm"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15GB RAM
GPU_TYPE="nvidia-tesla-t4"  # T4 GPU (good balance of cost/performance)
GPU_COUNT=1
IMAGE_FAMILY="pytorch-latest-gpu"  # Pre-configured PyTorch + CUDA image
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="100GB"

echo "=========================================="
echo "GCP GPU VM Setup for STT Project"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  VM Name: $VM_NAME"
echo "  Machine Type: $MACHINE_TYPE"
echo "  GPU: $GPU_TYPE x $GPU_COUNT"
echo "  Estimated Cost: ~\$0.35/hour"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk/install"
    exit 1
fi

# Set project
echo "📋 Setting GCP project..."
gcloud config set project $PROJECT_ID

# Check if VM already exists
if gcloud compute instances describe $VM_NAME --zone=$ZONE &>/dev/null; then
    echo "⚠️  VM '$VM_NAME' already exists in zone $ZONE"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing VM..."
        gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
    else
        echo "✅ Using existing VM. Run 'gcloud compute ssh $VM_NAME --zone=$ZONE' to connect."
        exit 0
    fi
fi

# Create GPU quota check (optional - may fail if quota not requested)
echo "🔍 Checking GPU quota..."
gcloud compute project-info describe --project=$PROJECT_ID | grep -q "nvidia-tesla-t4" || echo "⚠️  GPU quota may need to be requested"

# Create the VM
echo ""
echo "🚀 Creating GPU-enabled VM..."
echo "   This may take 5-10 minutes..."
echo ""

gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$DISK_SIZE \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --tags=http-server,https-server

echo ""
echo "✅ VM created successfully!"
echo ""
echo "📝 Next steps:"
echo "   1. SSH into the VM:"
echo "      gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "   2. On the VM, clone your repo and install dependencies:"
echo "      git clone <your-repo-url>"
echo "      cd Adaptive-Self-Learning-Agentic-AI-System"
echo "      pip install -r requirements.txt"
echo ""
echo "   3. Verify GPU access:"
echo "      python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "   4. Run your evaluation framework:"
echo "      python experiments/kavya_evaluation_framework.py"
echo ""
echo "💰 Cost Management:"
echo "   - Stop VM when not in use: gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "   - Start VM: gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "   - Delete VM: gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
echo "💡 Tip: Use preemptible instances for 60-80% cost savings (add --preemptible flag)"
echo ""

