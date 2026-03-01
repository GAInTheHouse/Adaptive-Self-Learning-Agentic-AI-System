# Migration Guide: Old → New Data Pipeline Scripts

**Effective Date:** February 27, 2026  
**Branch:** gxa/create-data  
**Status:** Deprecation warnings added, old scripts still functional

---

## Overview

The data pipeline has been redesigned in the `gxa/create-data` branch with a modular, local-first architecture. This guide helps migrate from old GCS-centric scripts to the new unified pipeline.

---

## Quick Migration Reference

| Old Script | Old Command | New Replacement | New Command |
|------------|-------------|-----------------|-------------|
| `download_datasets.py` | `python scripts/download_datasets.py` | `hf_download.py` | `python scripts/hf_download.py --dataset fsicoli/common_voice_17_0 --config en` |
| `download_datasets.py` | (automatic GCS upload) | `bootstrap_data.py` | `python scripts/bootstrap_data.py --download-hf` |
| `preprocess_data.py` | `python scripts/preprocess_data.py` | `augment_audio.py` | `python scripts/augment_audio.py --input-dir data/raw --output-dir data/augmented` |
| `preprocess_data.py` | (audio variants) | `bootstrap_data.py` | `python scripts/bootstrap_data.py --derive-low-audio --derive-corrupted` |

---

## Detailed Migration Examples

### Example 1: Download Common Voice

#### Old Way (download_datasets.py)
```bash
# Hardcoded to Common Voice 16.1, English, 5% sample
# Automatically uploads to GCS
python scripts/download_datasets.py
```

**Output:**
- `data/raw/common_voice_accents/` (HF format)
- `data/raw/common_voice_accents/metadata.json`
- Uploaded to `gs://stt-project-datasets/raw/common_voice_accents/`

#### New Way (hf_download.py)
```bash
# Generic, configurable, any HF dataset
python scripts/hf_download.py \
    --dataset fsicoli/common_voice_17_0 \
    --config en \
    --split train \
    --max-samples 5000 \
    --cache-dir data/hf_cache \
    --save-dir data/hf_saved \
    --audio-out-dir data/hf_audio \
    --manifest-dir data/manifests
```

**Output:**
- `data/hf_saved/fsicoli__common_voice_17_0__en/train/` (HF format)
- `data/hf_audio/fsicoli__common_voice_17_0__en/train/*.wav` (materialized audio)
- `data/manifests/fsicoli__common_voice_17_0__en__train.csv` (standardized manifest)

**Key Differences:**
- ✓ Not limited to 3 datasets
- ✓ Generates CSV manifests for training
- ✓ Materializes audio to WAV files
- ✗ No automatic GCS upload (manual if needed)

---

### Example 2: Download LibriSpeech

#### Old Way (download_datasets.py)
```bash
# Via HF API, only 10% of test-clean
python scripts/download_datasets.py
# Downloads: librispeech_asr dataset from HF
```

#### New Way (openslr_download.py or bootstrap_data.py)
```bash
# Direct from OpenSLR, full datasets
python scripts/openslr_download.py --dataset dev-clean
python scripts/openslr_download.py --dataset dev-other

# Or use orchestrator for multiple datasets
python scripts/bootstrap_data.py --download-openslr
```

**Output:**
- `data/openslr/SLR12_LibriSpeech/LibriSpeech/dev-clean/`
- `data/openslr/SLR12_LibriSpeech/LibriSpeech/dev-other/`
- Official OpenSLR directory structure (not HF format)

**Key Differences:**
- ✓ Direct from OpenSLR (faster, no HF API limits)
- ✓ Full datasets, not samples
- ✓ Checksum verification
- ✓ Resume support for interrupted downloads

---

### Example 3: Audio Preprocessing & Augmentation

#### Old Way (preprocess_data.py)
```bash
# Downloads from GCS, applies AudioPreprocessor, uploads back
python scripts/preprocess_data.py
```

**What it did:**
- Resampling to 16kHz (via librosa)
- Silence trimming (via librosa)
- Normalization (via librosa)
- Evaluation splits (80/10/10)
- Upload to GCS

#### New Way (augment_audio.py + bootstrap_data.py)

**Option A: Standalone augmentation**
```bash
python scripts/augment_audio.py \
    --input-dir data/openslr/SLR12_LibriSpeech/LibriSpeech/dev-clean \
    --musan-dir data/openslr/SLR17_MUSAN/musan/noise \
    --rirs-dir data/openslr/SLR28_RIRS_NOISES/RIRS_NOISES \
    --output-dir data/augmented \
    --snrs 0 5 10 15 20 \
    --reverb-probability 0.5 \
    --seed 1337
```

**Option B: Integrated workflow**
```bash
# Download datasets, generate manifests, create variants
python scripts/bootstrap_data.py --all

# Or step-by-step:
python scripts/bootstrap_data.py --download-openslr
python scripts/bootstrap_data.py --generate-manifests
python scripts/bootstrap_data.py --derive-low-audio     # 8kHz mono
python scripts/bootstrap_data.py --derive-corrupted     # MUSAN + RIRS augmentation
```

**Key Differences:**
- ✓ Uses canonical MUSAN/RIRS corpora (research standard)
- ✓ More augmentation types (noise, reverb, dropouts, clipping)
- ✓ Configurable SNR levels
- ✓ Generates CSV manifests for training
- ✗ No automatic evaluation splits (use src.data.evaluation_splits directly)
- ✗ No GCS integration (yet)

---

## Feature-by-Feature Migration

### Dataset Download Features

| Feature | Old Location | New Location | Notes |
|---------|-------------|--------------|-------|
| Common Voice download | `download_datasets.py` | `hf_download.py --dataset fsicoli/common_voice_17_0` | Now version 17.0, fully configurable (community mirror) |
| LibriSpeech download | `download_datasets.py` (via HF) | `openslr_download.py --dataset dev-clean` | Direct from OpenSLR, faster |
| Speech Commands | `download_datasets.py` | `hf_download.py --dataset google/speech_commands` | Generic HF support (official identifier) |
| Quality filtering | `download_datasets.py` (upvotes/downvotes) | **NOT MIGRATED** | Add to hf_download.py if needed |
| Accent filtering | `download_datasets.py` | **NOT MIGRATED** | Filter after download |
| Domain vocabulary | `download_datasets.py` | **NOT MIGRATED** | May not be needed |
| Dataset inventory | `download_datasets.py` | Implicit via manifests | Use manifests/ directory |
| GCS upload | `download_datasets.py` | **REMOVED** | Add manually if needed |

### Preprocessing Features

| Feature | Old Location | New Location | Notes |
|---------|-------------|--------------|-------|
| Resampling 16kHz | `preprocess_data.py` | `bootstrap_data.py` (via ffmpeg) | Faster ffmpeg vs librosa |
| Silence trimming | `src.data.preprocessing.py` | **NOT IN SCRIPTS** | Use module directly |
| Normalization | `src.data.preprocessing.py` | Implicit in augmentation | Part of augment_audio.py |
| Additive noise | **NONE** | `augment_audio.py` | New feature |
| Reverb (RIRS) | **NONE** | `augment_audio.py` | New feature |
| Random dropouts | **NONE** | `augment_audio.py` | New feature |
| Random clipping | **NONE** | `augment_audio.py` | New feature |
| Evaluation splits | `preprocess_data.py` | `src.data.evaluation_splits` | Still available as module |
| GCS download | `preprocess_data.py` | **REMOVED** | Stage data locally first |
| GCS upload | `preprocess_data.py` | **REMOVED** | Upload manually if needed |

---

## Workflow Migration

### Old Workflow (GCS-Centric)
```bash
# Step 1: Download datasets (uploads to GCS automatically)
python scripts/download_datasets.py

# Step 2: Preprocess (downloads from GCS, uploads results)
python scripts/preprocess_data.py

# Output: Everything in GCS
# gs://stt-project-datasets/raw/
# gs://stt-project-datasets/processed/
# gs://stt-project-datasets/evaluation/
```

### New Workflow (Local-First)
```bash
# Step 1: Download all datasets
python scripts/bootstrap_data.py --all

# OR step-by-step:
python scripts/bootstrap_data.py --download-openslr
python scripts/bootstrap_data.py --download-hf
python scripts/bootstrap_data.py --download-primock      # New: Medical consultations
python scripts/bootstrap_data.py --download-afrimedqa    # New: Medical QA

# Step 2: Generate manifests
python scripts/bootstrap_data.py --generate-manifests

# Step 3: Create audio variants
python scripts/bootstrap_data.py --derive-low-audio      # 8kHz mono
python scripts/bootstrap_data.py --derive-corrupted      # Augmented

# Step 4: (Optional) Upload to GCS manually
# gsutil -m cp -r data/manifests/ gs://your-bucket/manifests/
# gsutil -m cp -r data/openslr/ gs://your-bucket/openslr/
```

**Output:**
- Everything in `data/` directory locally
- CSV manifests in `data/manifests/` for training
- Multiple audio variants for robust training

---

## New Capabilities Not in Old Scripts

### 1. Medical Domain Datasets
```bash
# PriMock57: 57 medical consultations with transcripts
python scripts/primock_download.py

# AfriMed-QA: 15K medical questions (text-only)
python scripts/afrimedqa_download.py
```

### 2. Conversational Speech Datasets
```bash
# TED-LIUM Release 3: 430 hours of TED talks
python scripts/openslr_download.py --dataset tedlium3

# ST-AEDS: 4.7 hours of spontaneous speech
python scripts/openslr_download.py --dataset st_aeds
```

### 3. Standardized Manifest Format
All scripts generate unified CSV manifests:
```csv
dataset,split,utt_id,path,duration_seconds,sampling_rate,text,speaker,accent
librispeech,dev-clean,1272-128104-0000,/path/to/audio.flac,5.85,16000,"text",1272,
```

Use manifests for:
- Training data loaders
- Dataset statistics
- Quality filtering
- Cross-dataset comparison

### 4. Audio Materialization
Old scripts kept audio in HF dataset format. New scripts:
- Extract audio to WAV files
- Normalize paths for easy access
- Support both local and HF audio columns

---

## Addressing Missing Features from Old Scripts

### Feature: GCS Integration

**Old scripts:** Automatic upload/download  
**New scripts:** None yet

**Migration Options:**

**Option 1: Manual GCS Upload (Quick)**
```bash
# After running new pipeline, upload manually
gsutil -m cp -r data/manifests/ gs://stt-project-datasets/manifests/
gsutil -m cp -r data/openslr/ gs://stt-project-datasets/openslr/
gsutil -m cp -r data/hf_audio/ gs://stt-project-datasets/hf_audio/
```

**Option 2: Add GCS Flag to New Scripts (Recommended)**
```python
# Add to hf_download.py, openslr_download.py, etc.
if args.upload_gcs:
    from src.utils.gcs_utils import get_gcs_manager
    gcs_manager = get_gcs_manager("datasets")
    gcs_manager.upload_directory(str(output_dir), gcs_prefix)
```

**Option 3: Separate Upload Script**
```bash
# Create scripts/upload_to_gcs.py
python scripts/upload_to_gcs.py --local-dir data/manifests --gcs-prefix manifests/
```

### Feature: Dataset Filtering (Accent, Quality)

**Old scripts:** Built-in filtering for Common Voice  
**New scripts:** Download full datasets, filter separately

**Migration:**
```python
# Post-download filtering using pandas
import pandas as pd

manifest = pd.read_csv("data/manifests/common_voice__train.csv")

# Filter by accent
filtered = manifest[manifest['accent'].isin(['us', 'gb', 'au'])]

# Filter by duration (e.g., 3-10 seconds)
filtered = manifest[
    (manifest['duration_seconds'].astype(float) >= 3.0) &
    (manifest['duration_seconds'].astype(float) <= 10.0)
]

# Save filtered manifest
filtered.to_csv("data/manifests/common_voice__train_filtered.csv", index=False)
```

### Feature: Evaluation Splits

**Old scripts:** Created by `preprocess_data.py`  
**New scripts:** Use `src.data.evaluation_splits` module directly

**Migration:**
```python
# In your training script or notebook
from src.data.evaluation_splits import EvaluationSplitter

splitter = EvaluationSplitter(seed=42)
splits = splitter.create_splits(
    "data/manifests/librispeech__dev-clean.csv",
    "data/evaluation/librispeech",
    train_ratio=0.8,
    dev_ratio=0.1,
    test_ratio=0.1
)
```

---

## Architecture Philosophy Changes

### Old Architecture: Centralized & Cloud-First
- **Goal:** Production deployment with GCS
- **Design:** Tightly coupled to Google Cloud
- **Datasets:** Small samples (5-10%) for cost efficiency
- **Format:** Hugging Face datasets format
- **Workflow:** Download → GCS → Process → GCS

### New Architecture: Modular & Local-First
- **Goal:** Research flexibility with production scalability
- **Design:** Loosely coupled, composable scripts
- **Datasets:** Full datasets + medical/conversational focus
- **Format:** CSV manifests + materialized WAV audio
- **Workflow:** Download → Local → Manifests → Variants

**Why the change:**
1. **Research flexibility:** Easier to experiment locally
2. **No cloud lock-in:** Works without GCS account
3. **Reproducibility:** Anyone can run the pipeline
4. **Standardization:** CSV manifests are universal format
5. **Medical domain:** Added PriMock57 and AfriMed-QA datasets

---

## Code Migration Examples

### Migrating: Download Script

#### Old Code
```python
from src.utils.gcs_utils import get_gcs_manager

# Download
config = DATASETS_CONFIG["common_voice"]
dataset = load_dataset(
    config["name"],
    config["language"],
    split=config["split"],
    trust_remote_code=True
)

# Filter
dataset = dataset.filter(lambda ex: ex['up_votes'] > ex['down_votes'])

# Save
local_path = output_dir / "common_voice_accents"
dataset.save_to_disk(str(local_path))

# Upload to GCS
gcs_manager = get_gcs_manager("datasets")
gcs_manager.upload_directory(str(local_path), f"raw/{dataset_name}")
```

#### New Code
```python
# No GCS dependency needed
from datasets import load_dataset

dataset = load_dataset(
    "fsicoli/common_voice_17_0",
    "en",
    split="train",
)

# Filter after download (optional)
dataset = dataset.filter(lambda ex: ex.get('up_votes', 0) > ex.get('down_votes', 0))

# Save with save_to_disk
dataset.save_to_disk("data/hf_saved/common_voice_en/train")

# Generate manifest (built into hf_download.py)
# Or manually if needed using csv module
```

### Migrating: Preprocessing Script

#### Old Code
```python
from src.data.preprocessing import AudioPreprocessor
from src.data.evaluation_splits import EvaluationSplitter
from src.utils.gcs_utils import get_gcs_manager

# Download from GCS
gcs_manager = get_gcs_manager("datasets")
# ... download logic ...

# Preprocess
preprocessor = AudioPreprocessor(target_sr=16000, trim_silence=True, normalize=True)
# ... processing logic ...

# Create splits
splitter = EvaluationSplitter(seed=42)
splits = splitter.create_splits(...)

# Upload to GCS
gcs_manager.upload_directory(...)
```

#### New Code
```python
# Step 1: Augment audio locally
import subprocess

subprocess.run([
    "python", "scripts/augment_audio.py",
    "--input-dir", "data/openslr/SLR12_LibriSpeech/LibriSpeech/dev-clean",
    "--output-dir", "data/augmented/dev-clean",
    "--snrs", "0", "5", "10", "15", "20",
])

# Step 2: Create low-quality variants
subprocess.run([
    "python", "scripts/bootstrap_data.py",
    "--derive-low-audio",
])

# Step 3: Create evaluation splits (still use module)
from src.data.evaluation_splits import EvaluationSplitter

splitter = EvaluationSplitter(seed=42)
splits = splitter.create_splits(
    "data/manifests/librispeech__dev-clean.csv",
    "data/evaluation/librispeech",
    train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1
)
```

---

## Breaking Changes

### 1. No Automatic GCS Upload
**Impact:** Must manually upload or add GCS support back  
**Workaround:** Use `gsutil` commands or extend new scripts

### 2. Different Directory Structure
**Old:**
```
data/raw/common_voice_accents/
data/processed/common_voice_accents/
data/evaluation/common_voice_accents/
```

**New:**
```
data/hf_audio/fsicoli__common_voice_17_0__en/train/
data/manifests/fsicoli__common_voice_17_0__en__train.csv
data/derived/corrupted/fsicoli__common_voice_17_0__en/train/
```

**Impact:** Training scripts need updated paths  
**Workaround:** Use manifests (paths are absolute in CSV)

### 3. Manifest Format vs HF Dataset Format
**Old:** Data stayed in HF dataset format (accessible via `load_from_disk`)  
**New:** Data converted to CSV manifests + WAV files

**Impact:** Can't use `dataset = load_from_disk()` directly  
**Workaround:** Load manifests with pandas, audio with soundfile/librosa

```python
# Old way
from datasets import load_from_disk
dataset = load_from_disk("data/raw/common_voice_accents")
audio = dataset[0]['audio']

# New way
import pandas as pd
import librosa

manifest = pd.read_csv("data/manifests/common_voice__train.csv")
audio, sr = librosa.load(manifest.iloc[0]['path'])
```

---

## Testing Your Migration

### Verify Old Script Still Works
```bash
# Should show deprecation warning but still function
python scripts/download_datasets.py 2>&1 | head -30
```

### Test New Script Equivalents
```bash
# Download subset of Common Voice
python scripts/hf_download.py \
    --dataset fsicoli/common_voice_17_0 \
    --config en \
    --split validation \
    --max-samples 100 \
    --force

# Verify output
ls -lh data/manifests/
ls -lh data/hf_audio/fsicoli__common_voice_17_0__en/validation/
cat data/manifests/fsicoli__common_voice_17_0__en__validation.csv | head
```

### Compare Outputs
```python
# Quick comparison script
import pandas as pd

# Old manifest (if you generated one)
# old_meta = json.load(open("data/raw/common_voice_accents/metadata.json"))

# New manifest
new_manifest = pd.read_csv("data/manifests/fsicoli__common_voice_17_0__en__validation.csv")

print(f"New manifest samples: {len(new_manifest)}")
print(f"Columns: {list(new_manifest.columns)}")
print(new_manifest.head())
```

---

## Timeline

### Immediate (Current PR - gxa/create-data)
- ✅ New scripts fully functional
- ✅ Old scripts marked deprecated
- ✅ Documentation created (this guide)
- ⚠️ Both sets coexist

### Short Term (Next 1-2 PRs)
- Extract shared utilities to `scripts/dataset_utils.py`
- Refactor `bootstrap_data.py` to call external scripts
- Add optional GCS support to new scripts
- Update training scripts to use manifests

### Long Term (After validation period)
- Remove deprecated scripts (download_datasets.py, preprocess_data.py)
- Archive old workflow documentation
- Consolidate manifest generation if needed

---

## FAQ

### Q: Can I still use old scripts?
**A:** Yes, they still work but show deprecation warnings. Migrate when convenient.

### Q: Will old scripts be maintained?
**A:** No, bug fixes and features go to new scripts only.

### Q: What about GCS integration?
**A:** Removed for simplicity. Add back if needed or upload manually with `gsutil`.

### Q: How do I get evaluation splits?
**A:** Use `src.data.evaluation_splits.EvaluationSplitter` module directly in your training code.

### Q: Can I mix old and new workflows?
**A:** Yes, but not recommended. Stick to one approach per project.

### Q: What if I need dataset filtering?
**A:** Download full datasets, then filter using pandas on the manifest CSVs.

### Q: Which bootstrap flag replaces download_datasets.py?
**A:** `--download-hf` for HF datasets, but configure datasets manually first.

---

## Support & Questions

- **Documentation:** `scripts/README_DATASET_FETCHERS.md`
- **Script help:** `python scripts/<script_name>.py --help`
- **Duplicity analysis:** `scripts/DUPLICITY_ANALYSIS.md`
- **Test results:** `scripts/TEST_RESULTS.md`

---

## Appendix: Deprecated Script Warnings

Both deprecated scripts now display warnings on startup. Example:

```bash
$ python scripts/download_datasets.py

==============================================================
⚠️  DEPRECATION WARNING
==============================================================
This script (download_datasets.py) is deprecated.

Replacement: scripts/hf_download.py or scripts/bootstrap_data.py

See: scripts/MIGRATION_GUIDE.md for migration instructions
==============================================================

(Script continues to run...)
```
