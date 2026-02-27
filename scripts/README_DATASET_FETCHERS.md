# STT Dataset Fetchers

This directory contains scripts to download and prepare speech datasets for STT (Speech-to-Text) training.

## Overview

Three new dataset fetching scripts have been added to complement the existing data pipeline:

1. **`primock_download.py`** - Downloads PriMock57 medical consultation dataset
2. **`afrimedqa_download.py`** - Downloads AfriMed-QA medical QA dataset  
3. **`openslr_download.py`** - Enhanced to include TED-LIUM and ST-AEDS datasets
4. **`bootstrap_data.py`** - Updated to integrate all new downloaders

## Dataset Details

### 1. PriMock57 Medical Consultations

**Source:** https://github.com/babylonhealth/primock57

**Contents:**
- 57 mock medical primary care consultations
- Audio recordings (WAV format)
- Manual utterance-level transcriptions
- Consultation notes written by clinicians

**Requirements:**
- Git LFS must be installed (`brew install git-lfs` on macOS)
- ffprobe for audio metadata extraction

**Usage:**
```bash
# Download PriMock57 dataset
python scripts/primock_download.py

# Custom output directory
python scripts/primock_download.py --out-dir data/custom/primock57

# Force re-download
python scripts/primock_download.py --force

# Skip manifest generation (only clone repo)
python scripts/primock_download.py --skip-manifest
```

**Output:**
- Repository cloned to: `data/primock57/`
- Manifest generated: `data/manifests/primock57__full.csv`

### 2. AfriMed-QA Medical Question-Answering

**Source:** https://huggingface.co/datasets/afrimedqa/afrimedqa_v2

**Contents:**
- 15,000 medical questions (multiple-choice and open-ended)
- Questions from 60+ medical schools across 16 African countries
- Coverage of 32 medical specialties
- ACL 2025 Best Social Impact Paper Award

**Requirements:**
- `datasets` package (pip install datasets)
- `pandas` package (pip install pandas)

**Usage:**
```bash
# Download AfriMed-QA dataset
python scripts/afrimedqa_download.py

# Limit samples for testing
python scripts/afrimedqa_download.py --max-samples 100

# Custom output directory
python scripts/afrimedqa_download.py --out-dir data/custom/afrimedqa

# Force re-download
python scripts/afrimedqa_download.py --force
```

**Output:**
- Raw CSV data: `data/afrimedqa/afrimedqa_*.csv`
- Manifests: `data/manifests/afrimedqa__*.csv`

**Note:** This is a text-only dataset (no audio). Useful for:
- Understanding medical domain vocabulary
- Generating synthetic speech training data
- Evaluating medical domain ASR performance

### 3. OpenSLR Extended Datasets

**Enhanced datasets:**
- **TED-LIUM Release 3** (430 hours) - Conversational TED talks
- **ST-AEDS-20180100** (4.7 hours) - Spontaneous English speech

**Usage:**
```bash
# Download TED-LIUM Release 3
python scripts/openslr_download.py --dataset tedlium3

# Download ST-AEDS
python scripts/openslr_download.py --dataset st_aeds

# Download all OpenSLR datasets
python scripts/openslr_download.py --dataset all
```

**Output:**
- TED-LIUM: `data/openslr/SLR51_TEDLIUM/`
- ST-AEDS: `data/openslr/SLR45_STAEDS/`

## Integrated Bootstrap Workflow

The `bootstrap_data.py` script now includes all new datasets:

```bash
# Download everything (all datasets)
python scripts/bootstrap_data.py --all

# Download specific datasets
python scripts/bootstrap_data.py --download-primock
python scripts/bootstrap_data.py --download-afrimedqa
python scripts/bootstrap_data.py --download-openslr --include-tedlium --include-st-aeds

# Download PriMock57 and generate manifests
python scripts/bootstrap_data.py --download-primock --generate-manifests

# Check system dependencies
python scripts/bootstrap_data.py --check
```

### New Bootstrap Flags

- `--download-primock` - Download PriMock57 medical consultations
- `--download-afrimedqa` - Download AfriMed-QA medical QA dataset
- `--include-tedlium` - Include TED-LIUM Release 3 in OpenSLR downloads
- `--include-st-aeds` - Include ST-AEDS in OpenSLR downloads

## Data Directory Structure

After running the scripts, your data directory will be organized as:

```
data/
├── primock57/              # PriMock57 git repository
│   ├── audio/              # WAV recordings
│   ├── transcripts/        # Manual transcriptions
│   └── notes/              # Consultation notes
├── afrimedqa/              # AfriMed-QA CSV files
│   ├── afrimedqa_train.csv
│   ├── afrimedqa_test.csv
│   └── ...
├── openslr/                # OpenSLR datasets
│   ├── SLR12_LibriSpeech/
│   ├── SLR17_MUSAN/
│   ├── SLR28_RIRS_NOISES/
│   ├── SLR51_TEDLIUM/     # TED-LIUM Release 3
│   └── SLR45_STAEDS/      # ST-AEDS
├── manifests/              # Unified CSV manifests
│   ├── primock57__full.csv
│   ├── afrimedqa__train.csv
│   ├── afrimedqa__test.csv
│   └── ...
└── derived/                # Augmented audio variants
    ├── low_audio/
    └── corrupted/
```

## Manifest Format

All scripts generate manifests with a consistent CSV format:

**For audio datasets (PriMock57, LibriSpeech, etc.):**
```csv
dataset,split,utt_id,path,duration_seconds,sampling_rate,text,speaker,accent
```

**For text datasets (AfriMed-QA):**
```csv
dataset,split,utt_id,question_id,question_type,question,answer,specialty,country,difficulty,options,rationale
```

## Troubleshooting

### PriMock57 Download Issues

**Error:** "Git LFS is not installed"
```bash
# macOS
brew install git-lfs
git lfs install

# Ubuntu/Debian
sudo apt install git-lfs
git lfs install
```

**Error:** "ffprobe is not installed"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### AfriMed-QA Download Issues

**Error:** "Missing required package: datasets"
```bash
pip install datasets pandas
```

**Error:** Dataset access restricted
- Some Hugging Face datasets require accepting terms of use
- Visit the dataset page and agree to terms if prompted

## Citations

### PriMock57
```bibtex
@inproceedings{korfiatis2022primock57,
  title={PriMock57: A Dataset Of Primary Care Mock Consultations},
  author={Papadopoulos Korfiatis, Alex and Moramarco, Francesco and Sarac, Radmila and Savkov, Aleksandar},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

### AfriMed-QA
```bibtex
@inproceedings{nimo2025afrimed,
  title={AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset},
  author={Nimo, Charles and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```

## Next Steps

After downloading datasets:

1. **Generate Manifests:** Run manifest generation for all datasets
   ```bash
   python scripts/bootstrap_data.py --generate-manifests
   ```

2. **Create Audio Variants:** Generate low-quality and corrupted audio for robust training
   ```bash
   python scripts/bootstrap_data.py --derive-low-audio --derive-corrupted
   ```

3. **Train STT Models:** Use manifests to train speech recognition models with the collected data

## Support

For issues or questions:
- Check script help: `python scripts/<script_name>.py --help`
- Review error messages for specific guidance
- Ensure all prerequisites are installed
