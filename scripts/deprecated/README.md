# Deprecated Data Scripts

This folder contains data gathering scripts that have been superseded by the new unified modular data gathering system in `scripts/data_gatherer/`.

## Deprecation Notice

**As of February 2026**, all scripts in this folder are deprecated and should no longer be used. They have been replaced by a unified, plugin-based system that eliminates code duplication and provides a single entry point for all data operations.

## Migration Guide

### Old Scripts → New System

| Old Script | Lines | Replacement |
|-----------|-------|-------------|
| `download_datasets.py` | 338 | `scripts/data_gatherer/data_gather.py --sources huggingface` |
| `preprocess_data.py` | 194 | `scripts/augment_audio.py` + orchestrator |
| `bootstrap_data.py` | 1001 | `scripts/data_gatherer/data_gather.py` (200 lines) |
| `hf_download.py` | 348 | HuggingFacePlugin in new system |
| `openslr_download.py` | 300 | OpenSLRPlugin in new system |
| `primock_download.py` | 376 | GitPlugin in new system |
| `afrimedqa_download.py` | 270 | HuggingFacePlugin (text-only support) |
| `make_librispeech_manifest.py` | 169 | librispeech manifest generator |

**Total deprecated code: 3,796 lines**  
**New unified system: 1,652 lines (57% reduction)**

## Why Were These Deprecated?

### Code Duplication

- **937 lines of duplicated code** across 8 scripts
- 5 scripts duplicated manifest writing logic
- 3 scripts duplicated audio metadata extraction
- 2 scripts duplicated augmentation algorithms
- `bootstrap_data.py` reimplemented functionality from 3 other scripts

### Inconsistent Interfaces

- 3 different ways to download Common Voice
- Incompatible directory structures
- Different CLI argument patterns
- Scattered configuration across multiple files

### Maintenance Burden

- Bug fixes required changes in multiple files
- Adding new datasets required 50+ lines of code changes
- No single source of truth for dataset configuration

## New System Benefits

### Single Entry Point

```bash
# Download everything
python scripts/data_gatherer/data_gather.py --sources all

# Download specific sources
python scripts/data_gatherer/data_gather.py --sources huggingface openslr

# Download specific datasets
python scripts/data_gatherer/data_gather.py --datasets common_voice_17_0 tedlium3 primock57

# Convenience wrapper
python scripts/gather_data.py --sources all
```

### Plugin Architecture

- **HuggingFacePlugin**: Handles all 6 HF datasets (including text-only AfriMed-QA)
- **OpenSLRPlugin**: Handles all 8 OpenSLR datasets with resume support
- **GitPlugin**: Handles Git repos with LFS support (PriMock57)

### Declarative Configuration

All datasets defined in `scripts/data_gatherer/dataset_registry.yaml`:

- Add new dataset: 5 lines of YAML (vs 50+ lines of Python)
- Clear documentation of all data sources in one place
- Easy to version and track changes

### Zero Duplication

- Shared utilities in `dataset_utils.py`
- Each function exists in exactly one place
- Update once, benefit everywhere

## Data Sources Coverage

### Old System (3 datasets)

- Common Voice 16.1
- LibriSpeech ASR (HF)
- Speech Commands

### New System (15 datasets across 3 source types)

**Hugging Face (6):**
- Common Voice 16.1 & 17.0
- LibriSpeech ASR
- Speech Commands
- VoxPopuli
- AfriMed-QA (text-only)

**OpenSLR (8):**
- LibriSpeech dev-clean, dev-other, test-clean, test-other
- MUSAN noise corpus
- RIRS_NOISES
- TED-LIUM Release 3
- ST-AEDS

**Git (1):**
- PriMock57 medical consultations

## If You Need to Reference Old Code

The deprecated scripts remain in this folder for reference purposes only. Do not use them for new work.

For questions or issues with migration, see:
- `scripts/MIGRATION_GUIDE.md`
- `scripts/DUPLICITY_ANALYSIS.md`
- `scripts/CONSOLIDATION_RECOMMENDATIONS.md`

## Timeline

- **Before Feb 2026**: Multiple fragmented scripts with duplication
- **Feb 2026**: Unified system implemented in `scripts/data_gatherer/`
- **Current**: Old scripts deprecated and moved to this folder

---

**Use the new system**: `python scripts/data_gatherer/data_gather.py --help`
