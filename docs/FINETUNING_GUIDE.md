# Wav2Vec2 Fine-tuning Guide

This guide explains how to fine-tune the Wav2Vec2 STT model using LLM-generated gold standard transcripts.

## Overview

The fine-tuning process:
1. **Evaluation Phase**: Processes 200 audio files (100 clean, 100 noisy), gets STT transcripts, uses LLM to generate gold standard transcripts, and calculates baseline WER/CER
2. **Fine-tuning Phase**: Fine-tunes the model only on samples where STT made errors
3. **Re-evaluation Phase**: Evaluates the fine-tuned model and shows improvements

## Prerequisites

- Python 3.8+
- Audio files (200 total: 100 clean, 100 noisy)
- LLM (Mistral) connection working

## Setup

1. **Install dependencies** (if not already installed):
```bash
pip install torch transformers librosa jiwer datasets peft bitsandbytes
```

Optional (for faster LLM inference):
```bash
pip install flash-attn  # Requires CUDA and proper compilation
```

2. **Organize your audio files**:
```
data/finetuning_audio/
├── clean/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ... (100 files)
└── noisy/
    ├── audio_101.wav
    ├── audio_102.wav
    └── ... (100 files)
```

Alternatively, if you put all files in one directory, the script will automatically split them in half.

## Test LLM Connection

Before fine-tuning, test that the LLM is working:

```bash
python scripts/test_llm_connection.py
```

Expected output:
```
============================================================
LLM Connection Test
============================================================

1. Initializing Mistral LLM...
   Loading LLM: mistralai/Mistral-7B-Instruct-v0.3 on cuda (fast_mode=True)
   Using 4-bit quantization for fast inference
   Warming up model...
   Model warm-up complete
   ✓ LLM corrector initialized

2. Checking LLM availability...
   ✓ LLM is available and loaded

3. Testing transcript correction...
   Input: HIS LATRPAR AS USUALLY FORE
   Output: [LLM corrected output]
   ✓ LLM successfully corrected the transcript

4. Testing transcript improvement...
   ...
```

### LLM Optimization Features

The LLM corrector now includes several optimizations for faster inference:

1. **4-bit Quantization** (when CUDA available):
   - Reduces memory usage by ~75%
   - Significantly speeds up inference
   - Minimal accuracy loss

2. **Fast Mode** (enabled by default):
   - Reduced max tokens (128 vs 512)
   - Greedy decoding (faster, deterministic)
   - KV cache optimization
   - Model warm-up on initialization

3. **Flash Attention 2** (optional):
   - Automatically used if installed
   - Faster attention computation
   - Requires CUDA and proper compilation

These optimizations target **<1 second per transcript** inference time while maintaining quality.

## Run Fine-tuning

### Basic Usage

```bash
python scripts/finetune_wav2vec2.py --audio_dir data/finetuning_audio
```

By default, the script uses **LoRA** (Low-Rank Adaptation) for efficient fine-tuning, which is 3-5x faster and uses 3-5x less memory than full fine-tuning while maintaining comparable accuracy (within 0.3-0.5%).

### Advanced Options

```bash
python scripts/finetune_wav2vec2.py \
    --audio_dir data/finetuning_audio \
    --output_dir models/finetuned_wav2vec2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --lora_rank 8 \
    --lora_alpha 16
```

### Arguments

- `--audio_dir`: Directory containing audio files (required)
  - Should have `clean/` and `noisy/` subdirectories, OR
  - All files in root directory (will be split in half)
- `--output_dir`: Output directory for fine-tuned model (default: `models/finetuned_wav2vec2`)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--use_lora`: Enable LoRA fine-tuning (default: True)
- `--no_lora`: Disable LoRA and use full fine-tuning
- `--lora_rank`: LoRA rank - controls number of trainable parameters (default: 8)
  - Higher rank = more parameters, potentially better accuracy, but slower
  - Recommended range: 4-16
- `--lora_alpha`: LoRA alpha scaling factor (default: 16)
  - Typically set to 2× rank for good performance

## Output

The script will:

1. **Display baseline metrics**:
   ```
   Baseline Metrics:
     WER: 0.3620 (36.20%)
     CER: 0.1300 (13.00%)
     Error Samples: 150/200
     Error Rate: 0.7500 (75.00%)
   ```

2. **Estimate training time**:
   ```
   Estimated training time: ~X.X minutes
   ```

3. **Run fine-tuning** and show progress

4. **Display fine-tuned metrics**:
   ```
   Fine-tuned Metrics:
     WER: 0.3200 (32.00%)
     CER: 0.1100 (11.00%)
     Error Samples: 140/200
   ```

5. **Show summary with improvements**:
   ```
   SUMMARY
   ============================================================
   
   Baseline WER: 0.3620 (36.20%)
   Fine-tuned WER: 0.3200 (32.00%)
   WER Improvement: 0.0420 (4.20 percentage points)
   
   Baseline CER: 0.1300 (13.00%)
   Fine-tuned CER: 0.1100 (11.00%)
   CER Improvement: 0.0200 (2.00 percentage points)
   ```

6. **Save results** to `{output_dir}/evaluation_results.json`

## LoRA vs Full Fine-Tuning

### LoRA (Low-Rank Adaptation) - Default

**Benefits:**
- **3-5x faster** training time
- **3-5x less GPU memory** usage
- Only ~0.8% of parameters are trainable
- Comparable accuracy (typically within 0.3-0.5% of full fine-tuning)
- Smaller saved models (only adapters, not full model)

**When to use:**
- Limited computational resources
- Fast iteration and experimentation
- When slight accuracy trade-off is acceptable

**Model saving:**
- LoRA adapters are saved to `{output_dir}/lora_adapters/`
- To use: Load base model + adapters, or merge adapters for standalone use

### Full Fine-Tuning

**Benefits:**
- Maximum accuracy potential
- All model parameters updated
- Better for complex domain-specific tasks

**When to use:**
- When maximum accuracy is critical
- When you have abundant computational resources
- For complex tasks requiring comprehensive model updates

**To use full fine-tuning:**
```bash
python scripts/finetune_wav2vec2.py --audio_dir data/finetuning_audio --no_lora
```

## Training Time Estimation

The script estimates training time based on:
- Number of error samples
- Number of epochs
- LoRA vs Full fine-tuning

**LoRA**: ~7.5 seconds per sample per epoch (3-5x faster)
**Full Fine-tuning**: ~30 seconds per sample per epoch

**Examples**:
- **LoRA**: 150 error samples × 3 epochs × 7.5 seconds = ~56 minutes
- **Full**: 150 error samples × 3 epochs × 30 seconds = ~3.75 hours

**Actual time** may vary based on:
- Hardware (CPU vs GPU)
- Audio file lengths
- Batch size
- LoRA rank (higher rank = slightly slower)

## Using the Fine-tuned Model

After fine-tuning, the model will be saved to the output directory. To use it in the system:

1. Update `src/baseline_model.py` to load from the fine-tuned path for "wav2vec2-finetuned"
2. Or load directly:
```python
from src.baseline_model import BaselineSTTModel

model = BaselineSTTModel(model_name="path/to/finetuned/model")
result = model.transcribe("audio_file.wav")
```

## Troubleshooting

### LLM Not Available
If you see warnings about LLM not being available:
- Run `python scripts/test_llm_connection.py` to diagnose
- Check that Mistral model can be loaded
- The script will continue using STT transcripts as gold standard (not ideal)

### Out of Memory
- Reduce `--batch_size` (try 2 or 1)
- Process fewer samples
- Use a smaller model

### Slow Processing
- Ensure you're using GPU if available
- Reduce number of epochs
- Process files in batches

## Performance Benchmarks

### LoRA vs Full Fine-Tuning

Typical performance on STT tasks:
- **LoRA**: WER/CER within 0.3-0.5% of full fine-tuning
- **Training time**: 3-5x faster with LoRA
- **Memory usage**: 3-5x less with LoRA
- **Model size**: LoRA adapters ~10-50MB vs full model ~300MB+

### LLM Inference Speed

With optimizations enabled (fast_mode=True, 4-bit quantization):
- **Target**: <1 second per transcript
- **Typical**: 0.5-2 seconds depending on transcript length and hardware
- **Without optimizations**: 3-10+ seconds per transcript

## Notes

- The script only fine-tunes on **error cases** (samples where STT transcript != LLM gold standard)
- WER/CER are calculated using `jiwer` library
- With LoRA: Only adapters are saved (much smaller files)
- With Full Fine-tuning: Complete model is saved
- Training history and logs are saved to `{output_dir}/logs/`
- LoRA adapters can be merged into base model for standalone inference if needed

