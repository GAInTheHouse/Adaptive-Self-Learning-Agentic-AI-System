# GCP GPU Quick Start

## 🚀 Get Started in 3 Steps

### 1. Create GPU VM
```bash
bash scripts/setup_gcp_gpu.sh
```

### 2. Deploy & Run
```bash
python scripts/deploy_to_gcp.py
```

### 3. Monitor Costs
```bash
python scripts/monitor_gcp_costs.py
```

## 💡 Key Benefits

- **3-7x faster** inference on GPU vs CPU
- **Automatic GPU detection** - code works on both CPU/GPU
- **Cost monitoring** - track spending
- **Easy deployment** - one command to deploy

## 📖 Full Guide

See [docs/GCP_SETUP_GUIDE.md](docs/GCP_SETUP_GUIDE.md) for complete documentation.

## 💰 Cost Estimate

- **GPU VM**: ~$0.54/hour (~$12.96/day)
- **Storage**: ~$0.02/GB/month
- **Tip**: Stop VM when not in use to save costs!

## ✅ Quick Commands

```bash
# Create VM
bash scripts/setup_gcp_gpu.sh

# Deploy code
python scripts/deploy_to_gcp.py

# SSH into VM
gcloud compute ssh stt-gpu-vm --zone=us-central1-a

# Stop VM (save costs)
gcloud compute instances stop stt-gpu-vm --zone=us-central1-a

# Start VM
gcloud compute instances start stt-gpu-vm --zone=us-central1-a

# Check costs
python scripts/monitor_gcp_costs.py
```

