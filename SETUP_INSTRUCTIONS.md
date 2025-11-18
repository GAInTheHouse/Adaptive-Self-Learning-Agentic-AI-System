# GCP Setup Instructions - Step by Step

## 🎯 Overview

You need to install Google Cloud SDK (gcloud CLI) before you can create GPU VMs. Follow these steps:

## Step 1: Install Google Cloud SDK

### On macOS (Your System):

**Easiest Method - Using Homebrew:**
```bash
# Install Homebrew if needed (check: brew --version)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install gcloud CLI
brew install --cask google-cloud-sdk
```

**Alternative - Direct Install:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL  # Restart shell
```

## Step 2: Authenticate

```bash
# Login to your GCP account
gcloud auth login

# Set your project (use your actual project ID)
gcloud config set project stt-agentic-ai-2025
```

## Step 3: Enable Required APIs

```bash
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
```

## Step 4: Verify Setup

```bash
# Check installation
gcloud --version

# Check authentication
gcloud auth list

# Check project
gcloud config get-value project

# Or run the quick check script
bash scripts/quick_setup.sh
```

## Step 5: Create GPU VM

Once everything is set up:

```bash
# Create GPU VM (takes 5-10 minutes)
bash scripts/setup_gcp_gpu.sh
```

## Step 6: Deploy and Run

```bash
# Deploy code and run evaluation
python scripts/deploy_to_gcp.py
```

## 📋 Quick Checklist

- [ ] Install gcloud CLI
- [ ] Authenticate: `gcloud auth login`
- [ ] Set project: `gcloud config set project YOUR_PROJECT_ID`
- [ ] Enable APIs (compute, storage)
- [ ] Run: `bash scripts/quick_setup.sh` to verify
- [ ] Create VM: `bash scripts/setup_gcp_gpu.sh`
- [ ] Deploy: `python scripts/deploy_to_gcp.py`

## 🆘 Need Help?

1. **Installation issues**: See `scripts/INSTALL_GCLOUD.md`
2. **Setup questions**: See `docs/GCP_SETUP_GUIDE.md`
3. **Cost concerns**: Run `python scripts/monitor_gcp_costs.py`

## 💡 Tips

- **First time?** Start with the quick setup: `bash scripts/quick_setup.sh`
- **GPU quota?** You may need to request GPU quota in GCP Console
- **Costs?** Stop VMs when not in use: `gcloud compute instances stop stt-gpu-vm --zone=us-central1-a`

## 🚀 Ready?

Once gcloud is installed and authenticated, you're ready to create your GPU VM!

```bash
bash scripts/setup_gcp_gpu.sh
```

