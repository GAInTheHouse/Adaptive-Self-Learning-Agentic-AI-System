# Environment Setup

This document describes how to set up the development environment for the STT Autonomous Fine-Tuning Pipeline.

## Option 1: Conda Environment (Recommended for Development)

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Sufficient disk space (~10GB for environment + datasets)

### Quick Setup

```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate stt-genai

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
git lfs version
```

### What's Included

The `stt-genai` conda environment includes:

- **Python 3.9**
- **PyTorch 2.0+** with GPU support (if available)
- **Hugging Face libraries**: transformers, datasets, accelerate, peft
- **Audio processing**: librosa, soundfile, pydub, ffmpeg
- **Git LFS**: For downloading large file repositories (e.g., PriMock57)
- **Data science**: numpy, pandas, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Experiment tracking**: wandb
- **Development tools**: pytest, black, flake8

### Updating the Environment

If dependencies change:

```bash
# Update existing environment
conda env update -f environment.yml --prune

# Or recreate from scratch
conda env remove -n stt-genai
conda env create -f environment.yml
```

### Running Data Gathering Scripts

```bash
# Activate environment
conda activate stt-genai

# Download all datasets
python scripts/gather_data.py --sources all

# Download specific datasets
python scripts/gather_data.py --datasets common_voice_17_0 tedlium3
```

## Option 2: Docker (Recommended for Production)

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- Docker daemon running

### Build and Run

```bash
# Build the Docker image
docker build -t stt-api:latest .

# Run the container
docker run -p 8080:8080 \
  -e USE_GCS=false \
  -v $(pwd)/data:/app/data \
  stt-api:latest

# Or use docker-compose (if available)
docker-compose up
```

### What's Included

The Docker image includes:

- **Python 3.9** runtime
- **Git LFS** pre-installed and configured
- **FFmpeg** for audio processing
- **All Python dependencies** from requirements.txt
- **Production-ready** API server with health checks

### Data Gathering in Docker

To download datasets using Docker:

```bash
# Run data gathering inside container
docker run --rm \
  -v $(pwd)/data:/app/data \
  stt-api:latest \
  python scripts/gather_data.py --sources all
```

## Option 3: Manual Setup (Advanced)

If you prefer manual installation without conda or Docker:

### System Requirements

1. **Python 3.9+**
2. **Git LFS**:
   - macOS: `brew install git-lfs && git lfs install`
   - Ubuntu/Debian: `sudo apt install git-lfs && git lfs install`
   - Windows: Download from [git-lfs.github.com](https://git-lfs.github.com/)

3. **FFmpeg**:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Installation Steps

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
git lfs version
```

## Verification

After setup, verify everything works:

```bash
# Test import of key modules
python -c "
from scripts.data_gatherer.dataset_utils import configure_logging
from scripts.data_gatherer.source_plugins.huggingface_plugin import HuggingFacePlugin
from scripts.data_gatherer.source_plugins.openslr_plugin import OpenSLRPlugin
from scripts.data_gatherer.source_plugins.git_plugin import GitPlugin
print('✓ All modules imported successfully')
"

# Test data gathering with small dataset
python scripts/gather_data.py --datasets afrimedqa --force

# Check output
ls -lh data/manifests/afrimedqa*.csv
```

## Troubleshooting

### Issue: "Git LFS not found"

**Solution**:
- Conda: `conda install -c conda-forge git-lfs && git lfs install`
- Manual: See system requirements above

### Issue: VoxPopuli "Could not load libtorchcodec"

**Solution**:
- VoxPopuli requires torchcodec with FFmpeg shared libraries
- See detailed setup guide: `scripts/VOXPOPULI_SETUP.md`
- Quick fix (conda): `conda install -c conda-forge ffmpeg`

### Issue: "Failed to load dataset X"

**Common causes**:
1. Dataset removed from Hugging Face Hub → Check `scripts/data_gatherer/dataset_registry.yaml` for updated mirrors
2. Network connectivity issues → Check internet connection
3. Hugging Face authentication required → Run `huggingface-cli login`

### Issue: "ModuleNotFoundError"

**Solution**:
```bash
# Conda
conda env update -f environment.yml --prune

# Manual/Docker
pip install -r requirements.txt --upgrade
```

### Issue: Docker build fails

**Solution**:
- Ensure you have sufficient disk space (~5GB for image)
- Check Docker daemon is running: `docker info`
- Try clearing Docker cache: `docker system prune -a`

## Next Steps

After environment setup:

1. **Download datasets**: See `scripts/DATA_GATHERING_QUICKSTART.md`
2. **Train models**: See main `README.md`
3. **Run API**: See `src/README.md` (if available)

## Support

For issues or questions:
- Check documentation in `scripts/data_gatherer/README.md`
- Review `scripts/DATA_GATHERING_QUICKSTART.md` for common workflows
- Check `scripts/deprecated/README.md` for migration from old scripts
