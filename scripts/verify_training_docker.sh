#!/bin/bash
# Verification script for training Docker container
# Week 1 Task: Verify all training dependencies are available

set -e

echo "=========================================="
echo "Training Docker Environment Verification"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
echo ""

# Check PyTorch and CUDA
echo "2. Checking PyTorch and CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA not available. Training will use CPU (slower).')
"
echo ""

# Check Transformers
echo "3. Checking Transformers library..."
python3 -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
"
echo ""

# Check PEFT/LoRA
echo "4. Checking PEFT/LoRA support..."
python3 -c "
try:
    from peft import LoraConfig, get_peft_model, TaskType
    print('✓ PEFT/LoRA: Available')
    print('  LoRA support: Enabled')
except ImportError as e:
    print('✗ PEFT/LoRA: NOT AVAILABLE')
    print(f'  Error: {e}')
    exit(1)
"
echo ""

# Check Wav2Vec2
echo "5. Checking Wav2Vec2 support..."
python3 -c "
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print('✓ Wav2Vec2: Available')
    print('  Model loading: Supported')
except ImportError as e:
    print('✗ Wav2Vec2: NOT AVAILABLE')
    print(f'  Error: {e}')
    exit(1)
"
echo ""

# Check audio processing libraries
echo "6. Checking audio processing libraries..."
python3 -c "
import librosa
import soundfile
print(f'✓ Librosa version: {librosa.__version__}')
print('✓ SoundFile: Available')
"
echo ""

# Check evaluation metrics
echo "7. Checking evaluation metrics..."
python3 -c "
import jiwer
print('✓ jiwer: Available')
"
echo ""

# Check Google Cloud libraries
echo "8. Checking Google Cloud libraries..."
python3 -c "
try:
    from google.cloud import storage
    print('✓ Google Cloud Storage: Available')
except ImportError as e:
    print('⚠ Google Cloud Storage: Not available (optional for local training)')
"
echo ""

# Check data processing libraries
echo "9. Checking data processing libraries..."
python3 -c "
import pandas as pd
import numpy as np
import scipy
print(f'✓ Pandas version: {pd.__version__}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ SciPy version: {scipy.__version__}')
"
echo ""

# Check experiment tracking
echo "10. Checking experiment tracking..."
python3 -c "
try:
    import wandb
    print('✓ Weights & Biases: Available')
except ImportError:
    print('⚠ Weights & Biases: Not installed (optional)')
"
echo ""

# Test LoRA configuration
echo "11. Testing LoRA configuration..."
python3 -c "
from peft import LoraConfig, TaskType
config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    lora_dropout=0.1
)
print('✓ LoRA configuration: Valid')
print(f'  Rank: {config.r}')
print(f'  Alpha: {config.lora_alpha}')
"
echo ""

# Test Wav2Vec2 model loading (dry run)
echo "12. Testing Wav2Vec2 model loading (dry run)..."
python3 -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
print('Loading Wav2Vec2 processor...')
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
print('✓ Wav2Vec2 processor loaded successfully')
print('  Note: Full model loading skipped to save time')
"
echo ""

# Check training scripts
echo "13. Checking training scripts..."
if [ -f "/app/scripts/finetune_wav2vec2.py" ]; then
    echo "✓ finetune_wav2vec2.py: Found"
    python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from src.agent.fine_tuner import FineTuner
    print('✓ FineTuner class: Importable')
except ImportError as e:
    print(f'✗ FineTuner class: Import failed - {e}')
    exit(1)
"
else
    echo "✗ finetune_wav2vec2.py: NOT FOUND"
    exit(1)
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "✓ All core training dependencies verified"
echo "✓ LoRA support: Available"
echo "✓ Wav2Vec2 support: Available"
echo "✓ Training environment: Ready"
echo ""
echo "Next steps:"
echo "1. Test training with: python scripts/finetune_wav2vec2.py --help"
echo "2. Run smoke test: python scripts/finetune_wav2vec2.py --audio_dir <test_data> --num_epochs 1"
echo ""
