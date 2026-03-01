# VoxPopuli Dataset Setup Guide

VoxPopuli is a large-scale multilingual speech corpus (122GB, 182K+ examples for English) that requires special setup due to its dependency on `torchcodec` and FFmpeg shared libraries.

## Requirements

1. **torchcodec** - Python library for audio/video decoding (included in `requirements.txt` and `environment.yml`)
2. **FFmpeg shared libraries** - System libraries that torchcodec depends on

## Installation

### Option 1: Conda Environment (Recommended)

The FFmpeg package in conda-forge includes the necessary shared libraries:

```bash
# Activate the environment
conda activate stt-genai

# Install FFmpeg from conda-forge if not already installed
conda install -c conda-forge ffmpeg

# Verify torchcodec can load
python -c "import torchcodec; print('torchcodec ready')"
```

### Option 2: System FFmpeg (macOS)

Install FFmpeg via Homebrew with shared libraries:

```bash
# Install FFmpeg
brew install ffmpeg

# Set library path for torchcodec
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$DYLD_LIBRARY_PATH"

# Verify
python -c "import torchcodec; print('torchcodec ready')"
```

### Option 3: System FFmpeg (Linux)

Install FFmpeg development libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libavutil-dev libavcodec-dev libavformat-dev libswscale-dev

# Fedora/RHEL
sudo dnf install ffmpeg-devel

# Arch
sudo pacman -S ffmpeg

# Verify
python -c "import torchcodec; print('torchcodec ready')"
```

## Downloading VoxPopuli

Once torchcodec is properly configured:

```bash
# Download and generate manifests (this will take several hours)
python scripts/gather_data.py --datasets voxpopuli --force

# Check generated manifests
ls -lh data/manifests/voxpopuli*.csv
```

## Troubleshooting

### Error: "Could not load libtorchcodec"

**Cause**: FFmpeg shared libraries are not found.

**Solution**:
1. Ensure FFmpeg is installed with shared libraries
2. For conda: `conda install -c conda-forge ffmpeg`
3. For macOS Homebrew: Set `DYLD_LIBRARY_PATH` as shown above
4. For Linux: Install ffmpeg development packages

### Error: "Library not loaded: @rpath/libavutil.XX.dylib"

**Cause**: The FFmpeg version installed doesn't match what torchcodec expects.

**Solution**:
- torchcodec supports FFmpeg versions 4, 5, 6, 7, and 8
- Check your FFmpeg version: `ffmpeg -version`
- Install a compatible version via conda or system package manager

### Alternative: Skip VoxPopuli

If you don't need VoxPopuli, you can skip it:

```bash
# Download all datasets except VoxPopuli
python scripts/gather_data.py --sources huggingface --datasets common_voice_17_0 librispeech_asr speech_commands afrimedqa
```

## Dataset Information

- **Size**: ~122GB download, 182K+ English utterances
- **Source**: European Parliament event recordings (2009-2020)
- **Splits**: train, validation, test
- **Languages**: 18 languages available (English by default)
- **Use case**: ASR training for conversational/political speech

## Further Reading

- [VoxPopuli on Hugging Face](https://huggingface.co/datasets/facebook/voxpopuli)
- [TorchCodec Documentation](https://pytorch.org/torchcodec/)
- [FFmpeg Installation Guide](https://ffmpeg.org/download.html)
