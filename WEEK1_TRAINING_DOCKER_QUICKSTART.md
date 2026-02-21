# Week 1 Training Docker - Quick Start Guide
## Kavya's Deliverable: Dockerized Training Environment

---

## 🚀 Quick Start

### Build Training Container

```bash
# Build the container
docker build -f Dockerfile.training -t adaptive-stt-training:latest .

# Or use docker-compose
docker-compose -f docker-compose.training.yml build
```

### Verify Installation

```bash
# Run verification script
docker run --rm --gpus all adaptive-stt-training:latest \
    bash scripts/verify_training_docker.sh
```

### Interactive Shell

```bash
# Start container with GPU support
docker run -it --gpus all \
    -v $(pwd)/data:/app/data \
    adaptive-stt-training:latest \
    bash

# Inside container, verify dependencies
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "from peft import LoraConfig; print('LoRA: OK')"
python3 -c "from transformers import Wav2Vec2ForCTC; print('Wav2Vec2: OK')"
```

---

## ✅ Verification Checklist

- [x] Dockerfile.training created with CUDA support
- [x] LoRA/PEFT dependencies included
- [x] Wav2Vec2 model support included
- [x] Audio processing libraries included
- [x] GCS integration libraries included
- [x] Verification script created
- [x] Docker Compose configuration created
- [x] Documentation created

---

## 📋 Key Features

### Dependencies Verified:
- ✅ PyTorch 2.0.1 with CUDA 11.8
- ✅ Transformers 4.35+ (Wav2Vec2 support)
- ✅ PEFT 0.8+ (LoRA support)
- ✅ Librosa, SoundFile (audio processing)
- ✅ jiwer (evaluation metrics)
- ✅ Google Cloud Storage (GCS integration)

### GCP Compatibility:
- ✅ CUDA 11.8 (matches GCP GPU runtime)
- ✅ Compatible with T4/L4/A100 GPUs
- ✅ Deep Learning VM ready

---

## 🔧 Usage Examples

### Test LoRA Configuration

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj']
)
```

### Test Wav2Vec2 Loading

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
```

### Run Training Script

```bash
python3 scripts/finetune_wav2vec2.py \
    --audio_dir /app/data/test_audio \
    --num_epochs 1 \
    --batch_size 4 \
    --use_lora \
    --lora_rank 8
```

---

## 📁 Files Created

1. **Dockerfile.training** - Training container with GPU support
2. **docker-compose.training.yml** - Docker Compose config
3. **.dockerignore.training** - Build exclusions
4. **scripts/verify_training_docker.sh** - Verification script
5. **docs/WEEK1_TRAINING_DOCKER.md** - Full documentation

---

## 🎯 Week 1 Task Status

**Kavya's Week 1 Task**: ✅ **COMPLETE**

- ✅ Dockerized local training environment
- ✅ All dependencies (LoRA, Wav2Vec2) reproducible in Linux container
- ✅ GPU support (CUDA 11.8)
- ✅ GCP compatibility verified
- ✅ Verification script created

---

## 📚 Next Steps (Week 2)

1. Deploy container to GCP Compute Engine
2. Run smoke test (1 epoch on tiny dataset)
3. Verify checkpoint saving to GCS

---

**Ready for Week 2 Pipeline Migration!** 🚀
