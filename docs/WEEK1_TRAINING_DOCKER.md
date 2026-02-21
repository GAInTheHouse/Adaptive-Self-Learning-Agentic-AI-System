# Week 1: Training Docker Environment Setup
## Kavya's Deliverable: Dockerized Training Environment

**Status**: ✅ Complete  
**Date**: December 2024

---

## Overview

This document describes the Dockerized training environment for the Adaptive Self-Learning Agentic AI System. The container includes all dependencies for LoRA-based fine-tuning and Wav2Vec2 model training, ensuring reproducibility across different environments.

---

## Files Created

### 1. `Dockerfile.training`
- **Purpose**: Training-specific Dockerfile with GPU support
- **Base Image**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Key Features**:
  - CUDA 11.8 support (compatible with GCP T4/L4/A100 GPUs)
  - PyTorch with CUDA support
  - LoRA/PEFT library for parameter-efficient fine-tuning
  - Wav2Vec2 model support
  - All audio processing dependencies
  - GCS integration for data/model storage

### 2. `docker-compose.training.yml`
- **Purpose**: Docker Compose configuration for training environment
- **Features**:
  - GPU device passthrough
  - Volume mounts for data, checkpoints, models
  - Environment variable configuration
  - Network setup

### 3. `.dockerignore.training`
- **Purpose**: Exclude unnecessary files from Docker build context
- **Excludes**: Frontend, API files, test data, documentation

### 4. `scripts/verify_training_docker.sh`
- **Purpose**: Verification script to ensure all dependencies are available
- **Checks**: PyTorch, CUDA, LoRA, Wav2Vec2, audio libraries, GCS

---

## Building the Training Container

### Local Build

```bash
# Build the training container
docker build -f Dockerfile.training -t adaptive-stt-training:latest .

# Verify the build
docker run --rm --gpus all adaptive-stt-training:latest \
    bash scripts/verify_training_docker.sh
```

### Using Docker Compose

```bash
# Build and start training container
docker-compose -f docker-compose.training.yml build
docker-compose -f docker-compose.training.yml up -d

# Access the container
docker-compose -f docker-compose.training.yml exec training bash

# Run verification
docker-compose -f docker-compose.training.yml exec training \
    bash scripts/verify_training_docker.sh
```

---

## Key Dependencies Included

### Core ML Libraries
- **PyTorch 2.0.1** with CUDA 11.8 support
- **Transformers 4.35+** for model loading
- **Accelerate** for distributed training
- **Datasets** for data loading

### LoRA Support
- **PEFT 0.8+** for parameter-efficient fine-tuning
- **BitsAndBytes** for quantization support
- LoRA configuration for Wav2Vec2 attention modules

### Wav2Vec2 Support
- **Wav2Vec2ForCTC** model class
- **Wav2Vec2Processor** for audio preprocessing
- Support for `facebook/wav2vec2-base-960h` and variants

### Audio Processing
- **Librosa 0.10+** for audio analysis
- **SoundFile** for audio I/O
- **FFmpeg** and **SoX** for audio conversion

### Evaluation
- **jiwer** for WER/CER calculation

### Cloud Integration
- **Google Cloud Storage** for data/model storage
- **GCSFS** for filesystem-like access

---

## Verification Checklist

Run the verification script to ensure all dependencies are available:

```bash
bash scripts/verify_training_docker.sh
```

**Expected Output:**
- ✓ PyTorch with CUDA support
- ✓ LoRA/PEFT library available
- ✓ Wav2Vec2 models loadable
- ✓ Audio processing libraries
- ✓ Training scripts importable

---

## Testing the Training Environment

### 1. Quick Smoke Test

```bash
# Test LoRA configuration
python3 -c "
from peft import LoraConfig, TaskType
config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
)
print('LoRA config valid')
"

# Test Wav2Vec2 loading
python3 -c "
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
print('Wav2Vec2 processor loaded')
"
```

### 2. Test Fine-Tuning Script

```bash
# Check script help
python3 scripts/finetune_wav2vec2.py --help

# Run minimal training test (requires test data)
python3 scripts/finetune_wav2vec2.py \
    --audio_dir data/test_audio \
    --num_epochs 1 \
    --batch_size 2 \
    --use_lora
```

---

## GCP Compatibility

### Deep Learning VM Compatibility

The Dockerfile is designed to work with GCP Deep Learning VM images:
- **CUDA 11.8** matches GCP GPU runtime
- **T4 GPU** support (16GB VRAM)
- **L4 GPU** support (24GB VRAM)
- **A100 GPU** support (40GB/80GB VRAM)

### Testing on GCP

```bash
# On GCP VM, build the container
docker build -f Dockerfile.training -t gcr.io/PROJECT_ID/adaptive-stt-training:latest .

# Push to GCR
docker push gcr.io/PROJECT_ID/adaptive-stt-training:latest

# Run on GCP with GPU
docker run --gpus all \
    -v /path/to/data:/app/data \
    gcr.io/PROJECT_ID/adaptive-stt-training:latest \
    python3 scripts/finetune_wav2vec2.py --audio_dir /app/data
```

---

## Reproducibility Features

### 1. Fixed Versions
- PyTorch: 2.0.1 (CUDA 11.8)
- Transformers: >=4.35.0
- PEFT: >=0.8.0
- All dependencies pinned in `requirements.txt`

### 2. Deterministic Training
- Seed setting in training scripts
- Fixed CUDA operations (if applicable)
- Reproducible data loading

### 3. Environment Isolation
- All dependencies in container
- No host system dependencies
- Consistent Python version (3.10)

---

## Volume Mounts

The training container expects these directories:

```
/app/data/
├── raw/              # Raw audio files
├── processed/        # Processed audio
├── finetuning/      # Training datasets
├── checkpoints/      # Model checkpoints
├── models/          # Saved models
└── logs/           # Training logs
```

---

## Environment Variables

- `PYTHONUNBUFFERED=1`: Immediate output
- `PYTHONPATH=/app`: Source code path
- `CUDA_VISIBLE_DEVICES=0`: GPU selection
- `WANDB_API_KEY`: Experiment tracking (optional)
- `GOOGLE_APPLICATION_CREDENTIALS`: GCS access (optional)

---

## Troubleshooting

### CUDA Not Available
```bash
# Check GPU access
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### LoRA Import Errors
```bash
# Reinstall PEFT
pip3 install --upgrade peft
```

### Wav2Vec2 Loading Issues
```bash
# Test model loading
python3 -c "
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
print('Model loaded successfully')
"
```

---

## Next Steps (Week 2)

1. **GCP Deployment**: Deploy container to GCP Compute Engine
2. **Smoke Test**: Run 1-epoch training on GCP
3. **GCS Integration**: Verify checkpoint saving to GCS

---

## References

- [Dockerfile.training](../Dockerfile.training)
- [docker-compose.training.yml](../docker-compose.training.yml)
- [verify_training_docker.sh](../scripts/verify_training_docker.sh)
- [Fine-Tuning Guide](../docs/FINETUNING_GUIDE.md)

---

**Status**: ✅ Week 1 Training Docker Environment Complete
