# Data Gathering Quick Start Guide

**NEW**: All data scripts have been consolidated into `scripts/data_gatherer/`

## TL;DR

```bash
# Download everything (15+ datasets)
python scripts/gather_data.py --sources all

# Download specific datasets
python scripts/gather_data.py --datasets common_voice_17_0 tedlium3 primock57
```

## What Changed?

### Before (Deprecated)

- 8 different scripts with overlapping functionality
- 3,796 lines of code with 60-70% duplication
- Inconsistent interfaces and directory structures
- Scattered configuration

### After (Current)

- Single unified system in `scripts/data_gatherer/`
- 1,070 lines of well-structured code (71% reduction)
- Plugin architecture for extensibility
- YAML-based configuration for all datasets

## Available Datasets

### Speech Datasets (Audio)

1. **Common Voice** (16.1 & 17.0) - Crowdsourced multi-accent speech
2. **LibriSpeech** - Clean and challenging audiobook recordings
3. **Speech Commands** - Keyword spotting (10h)
4. **VoxPopuli** - European Parliament speeches (requires special setup - see `VOXPOPULI_SETUP.md`)
5. **TED-LIUM Release 3** - Conversational talks (430h)
6. **ST-AEDS** - Spontaneous speech (4.7h)
7. **PriMock57** - Medical consultations (57 samples)

### Augmentation Resources

8. **MUSAN** - Music, speech, noise for augmentation
9. **RIRS_NOISES** - Room impulse responses

### Text Datasets

10. **AfriMed-QA** - Medical QA (15K questions)

## Common Workflows

### 1. Download Training Data

```bash
# Core speech datasets
python scripts/gather_data.py --datasets \
  common_voice_17_0 \
  librispeech_dev_clean \
  voxpopuli
```

### 2. Download Augmentation Resources

```bash
# Noise and reverb for augmentation
python scripts/gather_data.py --datasets musan rirs_noises
```

### 3. Download Medical Domain Data

```bash
# Medical consultation audio + QA text
python scripts/gather_data.py --datasets primock57 afrimedqa
```

### 4. Download Everything

```bash
# All 15 datasets
python scripts/gather_data.py --sources all
```

## CLI Reference

```
usage: data_gather.py [-h] [--registry REGISTRY]
                      [--sources {all,huggingface,openslr,git} [...]]
                      [--datasets [DATASETS ...]]
                      [--output-dir OUTPUT_DIR]
                      [--manifest-dir MANIFEST_DIR]
                      [--augment] [--derive-variants] [--force]

Options:
  --sources          Source types: all, huggingface, openslr, git
  --datasets         Specific dataset names from registry
  --output-dir       Base directory for downloaded data (default: data/)
  --manifest-dir     Directory for CSV manifests (default: data/manifests/)
  --force            Re-download existing datasets
  --augment          Generate augmented variants (requires MUSAN/RIRS)
  --derive-variants  Generate low-quality and corrupted variants
```

## Prerequisites

### System Dependencies

```bash
# macOS
brew install git-lfs ffmpeg
git lfs install

# Ubuntu/Debian
sudo apt install git-lfs ffmpeg
git lfs install
```

### Python Packages

```bash
# Install in your conda environment
conda activate stt-genai
pip install -r requirements.txt
```

Key packages:
- `pyyaml` - YAML parsing
- `datasets` - Hugging Face datasets
- `soundfile` - Audio I/O
- `numpy` - Array operations
- `tqdm` - Progress bars

## Troubleshooting

### "ModuleNotFoundError: No module named 'yaml'"

```bash
conda activate stt-genai
pip install pyyaml
```

### "Git LFS required but not available"

```bash
brew install git-lfs  # macOS
# or
sudo apt install git-lfs  # Ubuntu

# Then initialize
git lfs install
```

### "ffprobe not found"

```bash
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu
```

### Download Fails or Hangs

- **OpenSLR**: Downloads are resumable. Re-run the same command.
- **Hugging Face**: Check internet connection, HF may be rate-limiting
- **Git**: Ensure git-lfs is installed for PriMock57

### Manifest CSV is Empty

- Ensure dataset downloaded completely (check output directory)
- Verify audio files exist in expected locations
- Check logs for parsing errors

## Directory Structure

After running the data gatherer:

```
data/
├── huggingface/           # HF datasets
│   ├── common_voice_17_0/
│   ├── voxpopuli/
│   └── afrimedqa/
├── openslr/               # OpenSLR datasets
│   ├── librispeech_dev_clean/
│   ├── musan/
│   ├── tedlium3/
│   └── rirs_noises/
├── git/                   # Git repos
│   └── primock57/
└── manifests/             # CSV manifests
    ├── common_voice_17_0__train.csv
    ├── librispeech__dev-clean.csv
    └── ... (one per dataset-split)
```

## Next Steps After Download

1. **Verify Manifests**: Check CSV files in `data/manifests/`
2. **Audio Augmentation**: Use `scripts/augment_audio.py` for noise/reverb
3. **Training**: Use manifests with your STT training pipeline

## Advanced: Custom Registry

Create your own registry for private datasets:

```yaml
# my_registry.yaml
version: "1.0"

huggingface:
  my_private_dataset:
    dataset: "organization/my-dataset"
    config: "en"
    splits: ["train"]
    description: "My custom dataset"
```

Use custom registry:

```bash
python scripts/gather_data.py \
  --registry /path/to/my_registry.yaml \
  --datasets my_private_dataset
```

## Getting Help

```bash
# Main command help
python scripts/gather_data.py --help

# See full documentation
cat scripts/data_gatherer/README.md

# Check available datasets
cat scripts/data_gatherer/dataset_registry.yaml
```

## Old Scripts

Old data scripts have been moved to `scripts/deprecated/`. Do not use them. See `scripts/deprecated/README.md` for migration information.

---

**Ready to download?** `python scripts/gather_data.py --sources all`
