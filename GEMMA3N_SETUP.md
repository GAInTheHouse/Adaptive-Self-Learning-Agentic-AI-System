# Gemma 3n Integration Setup

This document explains how to set up and use Gemma 3n for STT transcription.

## Prerequisites

1. **Apple Silicon Mac** (M1/M2/M3) - MLX is optimized for Apple Silicon
2. **Python 3.8+**

## Installation

Install the required MLX dependencies:

```bash
pip install mlx>=0.19.0 mlx-lm>=0.19.0 mlx-vlm>=0.1.0
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Model Details

- **Model Name**: `gg-hf-gm/gemma-3n-E2B-it` (2B effective parameters)
- **Alternative**: `gg-hf-gm/gemma-3n-E4B-it` (4B effective parameters, better accuracy)
- **Framework**: MLX (Apple Silicon optimized)
- **Use Case**: Fine-tuned STT models (gemma-finetuned-v2/v3) for demonstrating improved performance after fine-tuning

## Usage

The Gemma 3n model is automatically used when you select "Gemma Fine-tuned v2" or "Gemma Fine-tuned v3" in the UI dropdown. These represent the improved models after fine-tuning.

### Manual Usage

```python
from src.gemma3n_model import Gemma3nSTTModel

# Initialize model
model = Gemma3nSTTModel(model_name="gg-hf-gm/gemma-3n-E2B-it")

# Transcribe audio
result = model.transcribe("path/to/audio.wav")
print(result["transcript"])
```

## Integration with BaselineSTTModel

The `BaselineSTTModel` class automatically detects when to use Gemma 3n:

- `gemma-base-v1` → Uses Whisper-tiny (PyTorch) - Base model with poor performance
- `gemma-finetuned-v2` → Uses Gemma 3n (MLX) - Improved after fine-tuning
- `gemma-finetuned-v3` → Uses Gemma 3n (MLX) - Best performance after fine-tuning
- `gemma-3n` → Uses Gemma 3n (MLX) - Direct Gemma 3n access
- Other models → Uses Whisper (PyTorch)

## Notes

1. **First Run**: The model will be downloaded from Hugging Face on first use (~2-4GB)
2. **Performance**: Gemma 3n runs efficiently on Apple Silicon using MLX
3. **Audio Support**: The exact audio processing API may vary based on mlx-vlm version. The implementation includes fallbacks.

## Troubleshooting

### MLX Not Available
If you see "MLX not available" errors:
- Ensure you're on Apple Silicon (M1/M2/M3 Mac)
- Install MLX: `pip install mlx mlx-lm mlx-vlm`
- Check Python version: `python --version` (should be 3.8+)

### Model Download Issues
If model download fails:
- Check internet connection
- Ensure sufficient disk space (~4GB for E2B, ~8GB for E4B)
- Try manual download from Hugging Face

### Audio Transcription Issues
If transcription doesn't work:
- Check that mlx-vlm supports audio input in your version
- The implementation includes fallbacks for different API versions
- Check logs for specific error messages

## Performance Comparison

| Model | Framework | Parameters | Speed (Apple Silicon) | Accuracy | Use Case |
|-------|-----------|------------|----------------------|----------|----------|
| Whisper Tiny | PyTorch | 39M | Fast | Lower | Base model (poor performance) |
| Gemma 3n E2B | MLX | 2B | Fast | Better | Fine-tuned v2 (improved) |
| Gemma 3n E4B | MLX | 4B | Medium | Best | Fine-tuned v3 (best) |
| Whisper Base | PyTorch | 74M | Medium | Good | Alternative fine-tuned option |

## Next Steps

For production use, you may want to:
1. Fine-tune Gemma 3n on your domain-specific data
2. Use Gemma 3n E4B for better accuracy
3. Optimize the audio processing pipeline
4. Add caching for faster repeated transcriptions

