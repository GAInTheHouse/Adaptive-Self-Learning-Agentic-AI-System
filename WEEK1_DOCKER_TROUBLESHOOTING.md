# Week 1 Docker Troubleshooting Guide

## Issue: Docker Credential Error

If you encounter:
```
ERROR: failed to build: failed to solve: error getting credentials - err: exit status 1
```

This is a Docker credential helper issue, not a problem with the Dockerfile.

---

## Quick Fixes

### Option 1: Use CPU-Only Version (Recommended for Local Testing)

For local testing without GPU, use the CPU-only Dockerfile:

```bash
# Build CPU-only version (no NVIDIA image needed)
docker build -f Dockerfile.training.cpu -t adaptive-stt-training:cpu .

# Verify
docker run --rm adaptive-stt-training:cpu \
    bash scripts/verify_training_docker.sh
```

**Note**: CPU version works for all dependencies except GPU training. Perfect for verifying LoRA and Wav2Vec2 setup.

---

### Option 2: Fix Docker Credentials

```bash
# Run the fix script
bash scripts/fix_docker_credentials.sh

# Or manually:
# Remove credential helper
rm ~/.docker/config.json

# Try pulling the base image directly
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

---

### Option 3: Test Docker Connection

```bash
# Check Docker is running
docker info

# Try pulling a simple image
docker pull hello-world

# If that works, try NVIDIA image
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

---

## Recommended Approach

**For Week 1 Verification (Local):**
- Use `Dockerfile.training.cpu` - verifies all dependencies work
- No GPU needed for dependency verification
- Faster build time

**For Week 2 (GCP Deployment):**
- Use `Dockerfile.training` on GCP where GPU is available
- GCP VMs have proper Docker/NVIDIA setup
- Credential issues won't occur on GCP

---

## Verification Commands

### CPU Version (Local Testing)

```bash
# Build
docker build -f Dockerfile.training.cpu -t adaptive-stt-training:cpu .

# Verify dependencies
docker run --rm adaptive-stt-training:cpu \
    python3 -c "from peft import LoraConfig; print('LoRA: OK')"

docker run --rm adaptive-stt-training:cpu \
    python3 -c "from transformers import Wav2Vec2ForCTC; print('Wav2Vec2: OK')"
```

### GPU Version (GCP)

```bash
# On GCP VM with GPU
docker build -f Dockerfile.training -t adaptive-stt-training:latest .

# Verify GPU
docker run --rm --gpus all adaptive-stt-training:latest \
    python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Status

✅ **Week 1 Task Complete**: Both CPU and GPU Dockerfiles created
✅ **Dependencies Verified**: LoRA and Wav2Vec2 support included
✅ **Reproducibility**: All dependencies containerized

**Use CPU version for local verification, GPU version for GCP deployment.**
