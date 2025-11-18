# GCP GPU Setup Guide

Complete guide for setting up and using Google Cloud Platform GPU resources for the STT project.

## 🎯 Overview

This guide helps you:
- Create GPU-enabled VMs on GCP
- Deploy and run your evaluation framework on GPU
- Monitor costs and optimize spending
- Leverage GCP credits effectively

## 📋 Prerequisites

1. **GCP Account** with credits
2. **gcloud CLI** installed: [Install Guide](https://cloud.google.com/sdk/install)
3. **Project ID**: `stt-agentic-ai-2025` (or update in scripts)

## 🚀 Quick Start

### Step 1: Create GPU VM

```bash
# Make script executable
chmod +x scripts/setup_gcp_gpu.sh

# Run setup script
bash scripts/setup_gcp_gpu.sh
```

This will:
- Create a VM with NVIDIA T4 GPU
- Install PyTorch + CUDA
- Configure GPU drivers
- Set up project access

**Estimated Cost**: ~$0.54/hour (~$12.96/day if running 24/7)

### Step 2: Deploy Code to VM

```bash
# Deploy code and run evaluation
python scripts/deploy_to_gcp.py
```

This will:
- Upload your code to the VM
- Install dependencies
- Verify GPU access
- Run evaluation framework
- Download results

### Step 3: Monitor Costs

```bash
# Check VM status and costs
python scripts/monitor_gcp_costs.py
```

## 💻 Manual Setup (Alternative)

If you prefer manual setup:

### 1. Create VM Manually

```bash
gcloud compute instances create stt-gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### 2. SSH into VM

```bash
gcloud compute ssh stt-gpu-vm --zone=us-central1-a
```

### 3. On VM: Clone and Setup

```bash
# Clone your repository
git clone <your-repo-url>
cd Adaptive-Self-Learning-Agentic-AI-System

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. Run Evaluation

```bash
# Run evaluation framework (will automatically use GPU)
python experiments/kavya_evaluation_framework.py
```

## 📊 GPU vs CPU Performance

Based on your benchmarks:

| Metric | CPU | GPU (T4) | Speedup |
|--------|-----|----------|---------|
| Latency (per sample) | 0.72s | ~0.1-0.2s | **3-7x faster** |
| Throughput | 2.97 samples/s | ~10-20 samples/s | **3-7x faster** |
| 100 samples | 72s | 10-20s | **3-7x faster** |

## 💰 Cost Management

### VM Costs

- **T4 GPU + n1-standard-4**: ~$0.54/hour
- **Per day (24h)**: ~$12.96
- **Per month (730h)**: ~$394.20

### Cost-Saving Tips

1. **Stop VMs when not in use**:
   ```bash
   gcloud compute instances stop stt-gpu-vm --zone=us-central1-a
   gcloud compute instances start stt-gpu-vm --zone=us-central1-a
   ```

2. **Use Preemptible Instances** (60-80% cheaper):
   ```bash
   # Add --preemptible flag when creating VM
   ```

3. **Use Smaller GPUs for Development**:
   - T4: Good for development (~$0.35/hour GPU)
   - V100: For training (~$2.50/hour GPU)
   - A100: For heavy training (~$3.00/hour GPU)

4. **Set Billing Alerts**:
   - Go to GCP Console → Billing → Budgets & Alerts
   - Set up alerts at 50%, 90%, 100% of budget

### Storage Costs

- **GCS Storage**: ~$0.02/GB/month
- **Current buckets**:
  - `stt-project-datasets`: Store datasets
  - `stt-project-models`: Store model checkpoints
  - `stt-project-logs`: Store training logs

## 🔧 GPU Optimization

The code automatically optimizes for GPU when available:

- **TensorFloat-32 (TF32)**: Enabled for Ampere+ GPUs
- **Beam Search**: Better quality with GPU
- **KV Cache**: Faster generation on GPU
- **Half Precision**: Can be enabled for 2x speedup (with minor quality loss)

## 📁 File Structure

```
scripts/
├── setup_gcp_gpu.sh          # Create GPU VM
├── deploy_to_gcp.py          # Deploy and run on VM
└── monitor_gcp_costs.py      # Monitor costs

docs/
└── GCP_SETUP_GUIDE.md        # This guide
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# On VM, check NVIDIA drivers
nvidia-smi

# If not available, install drivers
sudo /opt/deeplearning/install-driver.sh
```

### Out of Memory

- Use smaller batch sizes
- Use gradient checkpointing
- Use smaller models (whisper-tiny instead of whisper-base)

### VM Won't Start

- Check GPU quota: `gcloud compute project-info describe`
- Request quota increase if needed
- Try different zone

### High Costs

- Stop VM immediately: `gcloud compute instances stop stt-gpu-vm`
- Check running VMs: `gcloud compute instances list`
- Review billing: https://console.cloud.google.com/billing

## 📚 Additional Resources

- [GCP GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- [PyTorch on GCP](https://cloud.google.com/ai-platform/training/docs/getting-started-pytorch)
- [Cost Calculator](https://cloud.google.com/products/calculator)
- [Preemptible VMs](https://cloud.google.com/compute/docs/instances/preemptible)

## ✅ Checklist

- [ ] GCP account with credits
- [ ] gcloud CLI installed and authenticated
- [ ] GPU VM created
- [ ] Code deployed to VM
- [ ] GPU verified working
- [ ] Evaluation framework runs on GPU
- [ ] Billing alerts configured
- [ ] VM stopped when not in use

## 🎓 Next Steps

1. **Run full evaluation on GPU** - See 3-7x speedup
2. **Train/fine-tune models** - Use GPU for training
3. **Scale up evaluation** - Run on larger datasets
4. **Deploy API** - Use Cloud Run for inference API

---

**Need Help?** Check the scripts or run with `--help` flag for options.

