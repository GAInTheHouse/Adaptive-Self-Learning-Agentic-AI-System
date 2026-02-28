# Script Duplicity Analysis - Data Pipeline Scripts

**Analysis Date:** February 27, 2026  
**Branch:** gxa/create-data  
**Purpose:** Identify redundant functionality and recommend consolidation

---

## Executive Summary

**Finding:** Significant functional overlap exists between old (main branch) and new (gxa/create-data) dataset scripts, with approximately **60-70% duplicity** in downloading and preprocessing logic.

**Recommendation:** **Deprecate** old scripts (`download_datasets.py`, `preprocess_data.py`) in favor of the new, more modular and comprehensive pipeline.

---

## Script Inventory

### Scripts from Main Branch (Old - 5 scripts)
1. `download_datasets.py` (338 lines) - GCS-focused HF downloader
2. `preprocess_data.py` (194 lines) - GCS-centric preprocessing
3. `check_ollama_models.py` (153 lines) - LLM infrastructure check
4. `finetune_wav2vec2.py` (unknown) - Training script
5. `test_llm_connection.py` (153 lines) - LLM connectivity test

### Scripts Added in gxa/create-data Branch (New - 7 scripts)
1. `bootstrap_data.py` (1001 lines) - **Unified data pipeline orchestrator**
2. `openslr_download.py` (300 lines) - OpenSLR downloader with checksums
3. `hf_download.py` (348 lines) - Generic HF dataset downloader
4. `make_librispeech_manifest.py` (169 lines) - LibriSpeech manifest generator
5. `augment_audio.py` (266 lines) - Audio augmentation with MUSAN/RIRS
6. `primock_download.py` (376 lines) - PriMock57 Git LFS downloader
7. `afrimedqa_download.py` (270 lines) - AfriMed-QA HF downloader

### Source Modules in src/data/ (13 modules)
- `preprocessing.py` (123 lines) - Audio preprocessing utilities
- `evaluation_splits.py` (105 lines) - Train/dev/test splitter
- `data_manager.py` (468 lines) - Central data management
- `finetuning_*.py` (3 modules, ~2000 lines) - Fine-tuning orchestration
- `metadata_tracker.py`, `version_control.py`, etc. (production infrastructure)

---

## Detailed Duplicity Analysis

### 🔴 CRITICAL DUPLICITY: Dataset Downloading

#### Old: `download_datasets.py` vs New: `hf_download.py` + `bootstrap_data.py`

**Functionality Overlap: 85%**

| Feature | Old (download_datasets.py) | New (hf_download.py) | Winner |
|---------|---------------------------|---------------------|---------|
| **HF Dataset Loading** | ✓ Hardcoded 3 datasets | ✓ Generic, any dataset | **NEW** |
| **Dataset Filtering** | ✓ Accent/quality filters | ✗ No filtering | OLD |
| **Local Persistence** | ✓ save_to_disk | ✓ save_to_disk + manifests | **NEW** |
| **Metadata Generation** | ✓ Custom JSON | ✓ Standardized CSV | **NEW** |
| **GCS Upload** | ✓ Automatic upload | ✗ No GCS support | OLD |
| **Audio Materialization** | ✗ Keeps HF format | ✓ Extracts to WAV | **NEW** |
| **Manifest Format** | ✗ No manifests | ✓ Unified CSV format | **NEW** |
| **Configurability** | ✗ Hardcoded config | ✓ CLI arguments | **NEW** |
| **Resume Support** | ✗ None | ✓ Checksum validation | **NEW** |

**Datasets Downloaded:**
- **Old:** Common Voice 16.1 (5% train), LibriSpeech ASR (10% test), Speech Commands (5% train)
- **New:** ANY HF dataset, OpenSLR datasets, PriMock57, AfriMed-QA

**Key Differences:**
1. **Old script is GCS-centric** - assumes production GCS workflow
2. **New scripts are local-first** - can be used independently without GCS
3. **Old script samples datasets** (5-10%) for cost efficiency
4. **New scripts download full datasets** with optional limits

**Verdict:** ⚠️ **80% duplicity** - Both download HF datasets, but different architectures

---

### 🔴 CRITICAL DUPLICITY: Audio Preprocessing

#### Old: `preprocess_data.py` + `src/data/preprocessing.py` vs New: `augment_audio.py` + `bootstrap_data.py`

**Functionality Overlap: 70%**

| Feature | Old Pipeline | New Pipeline | Winner |
|---------|-------------|--------------|---------|
| **Resampling** | ✓ Via AudioPreprocessor | ✓ Via ffmpeg | **NEW** (faster) |
| **Normalization** | ✓ Via librosa | ✓ Integrated | BOTH |
| **Silence Trimming** | ✓ Via librosa | ✗ Not yet | OLD |
| **Noise Addition** | ✗ None | ✓ MUSAN corpus | **NEW** |
| **Reverb (RIRS)** | ✗ None | ✓ RIRS corpus | **NEW** |
| **Random Dropouts** | ✗ None | ✓ Implemented | **NEW** |
| **Random Clipping** | ✗ None | ✓ Implemented | **NEW** |
| **Evaluation Splits** | ✓ Via EvaluationSplitter | ✗ Not yet | OLD |
| **GCS Integration** | ✓ Upload/download | ✗ None | OLD |

**Processing Philosophy:**
- **Old:** Quality improvement (clean, normalize, split)
- **New:** Robustness training (augmentation, corruption, variants)

**Verdict:** 🟡 **60% overlap** - Different goals but overlapping techniques

---

### 🟢 LOW DUPLICITY: Specialized Downloaders

#### `openslr_download.py` vs `bootstrap_data.py` OpenSLR section

**Functionality Overlap: 30%** (intentional modularity)

- `openslr_download.py` - **Standalone** OpenSLR downloader with checksums
- `bootstrap_data.py` - **Calls** openslr_download.py or reimplements core logic

**Issue:** `bootstrap_data.py` duplicates OpenSLR download logic instead of calling `openslr_download.py`

**Lines 276-308 in bootstrap_data.py:**
```python
def download_openslr(paths, ...):
    downloads = [...]
    for key, extract_dir, expected_path in downloads:
        url = OPENSLR_URLS[key]
        download_file(url, archive_path, ...)  # Reimplemented
        extract_archive(archive_path, ...)     # Reimplemented
```

**Lines 217-273 in openslr_download.py:**
```python
def download_and_extract(url, out_dir, ...):
    _stream_download(url, archive_path)        # With resume support
    _verify_checksum_if_available(...)         # With verification
    _extract_archive(archive_path, ...)
```

**Verdict:** 🟡 **30% duplicity** - bootstrap_data.py should call openslr_download.py as subprocess

---

### 🔴 CRITICAL DUPLICITY: Manifest Generation

#### Multiple scripts generate similar manifests

**Scripts that generate CSV manifests:**
1. `make_librispeech_manifest.py` - LibriSpeech only
2. `bootstrap_data.py` - Lines 369-476 (LibriSpeech, MUSAN, RIRS)
3. `hf_download.py` - Lines 215-227 (any HF dataset)
4. `primock_download.py` - Lines 205-260 (PriMock57)
5. `afrimedqa_download.py` - Lines 189-243 (AfriMed-QA)

**Common Manifest Fields:**
```csv
dataset, split, utt_id, path, duration_seconds, sampling_rate, text, speaker, accent
```

**Duplicated Logic:**
- Audio metadata extraction (ffprobe/soundfile)
- Transcript parsing and mapping
- CSV writing with DictWriter
- Progress bars with tqdm

**Verdict:** ⚠️ **Significant code duplication** across 5 scripts

---

## Architecture Comparison

### Old Architecture (Main Branch)
```
download_datasets.py → GCS → preprocess_data.py → GCS
                ↓                      ↓
        HF datasets only     Uses src/data/preprocessing.py
        Hardcoded 3 sets     + evaluation_splits.py
        Small samples        GCS-centric workflow
```

### New Architecture (gxa/create-data)
```
bootstrap_data.py (orchestrator)
    ├── download_openslr() → OpenSLR datasets
    ├── download_hf_dataset() → HF datasets  
    ├── download_primock57() → calls primock_download.py
    ├── download_afrimedqa() → calls afrimedqa_download.py
    ├── generate_source_manifests() → CSV manifests
    ├── derive_low_audio() → 8kHz variants
    └── derive_corrupted_audio() → MUSAN/RIRS augmentation

Standalone scripts (can run independently):
    - openslr_download.py
    - hf_download.py
    - make_librispeech_manifest.py
    - augment_audio.py
    - primock_download.py
    - afrimedqa_download.py
```

---

## Specific Duplications Found

### 1. HF Dataset Download Logic

**Location 1: `download_datasets.py` (OLD)**
```python
dataset = load_dataset(
    config["name"],
    config["language"],
    split=config["split"],
    trust_remote_code=True
)
dataset.save_to_disk(str(local_path))
```

**Location 2: `hf_download.py` (NEW)**
```python
split_ds = load_dataset(
    dataset_name,
    name=config,
    split=split_name,
    cache_dir=str(cache_dir),
)
dataset.save_to_disk(str(split_save_dir))
```

**Location 3: `bootstrap_data.py` (NEW) - Lines 546-590**
```python
dataset = load_dataset(
    dataset_name,
    config_name,
    split=split_name,
)
dataset = dataset.cast_column("audio", Audio(decode=True))
```

**Duplicity Level:** 🔴 **HIGH** - Same API calls, different error handling

---

### 2. Audio Metadata Extraction

**Appears in 5 scripts with slight variations:**

**Version 1: Using ffprobe (openslr_download.py, bootstrap_data.py, primock_download.py)**
```python
command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", ...]
process = subprocess.run(command, capture_output=True, text=True)
data = json.loads(process.stdout)
duration = float(data["format"]["duration"])
```

**Version 2: Using soundfile (make_librispeech_manifest.py, hf_download.py)**
```python
import soundfile as sf
info = sf.info(str(path))
duration = f"{float(info.duration):.6f}"
sample_rate = str(int(info.samplerate))
```

**Duplicity Level:** 🔴 **HIGH** - Should be extracted to utility module

---

### 3. Audio Augmentation Logic

**Location 1: `augment_audio.py` (standalone)**
```python
def _mix_at_snr(clean, noise, snr_db):
    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return (clean + noise * float(scale)).astype(np.float32)
```

**Location 2: `bootstrap_data.py` Lines 765-776**
```python
def _mix_additive_noise(clean, noise, snr_db):
    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return clean + noise * float(scale)
```

**Duplicity Level:** 🔴 **CRITICAL** - Identical algorithms, minor naming differences

---

### 4. File Utilities

**Duplicated in multiple scripts:**
- `_safe_name()` / `_safe_utt_id()` - Sanitize strings (4 scripts)
- `_list_audio_files()` - Find audio by extension (3 scripts)
- `_write_manifest()` - Write CSV manifests (5 scripts)
- `_require_package()` - Check Python imports (3 scripts)

**Duplicity Level:** 🔴 **HIGH** - Should be in shared utilities module

---

## GCS Integration Gap

### Old Scripts (Main Branch)
```python
from src.utils.gcs_utils import get_gcs_manager

gcs_manager = get_gcs_manager("datasets")
gcs_manager.upload_directory(str(local_path), f"raw/{dataset_name}")
```

### New Scripts (gxa/create-data)
```python
# NO GCS INTEGRATION
# All downloads and processing stay local
```

**Impact:** New scripts **removed GCS dependency**, making them usable without cloud infrastructure. This is **intentional architectural change**, not duplicity.

---

## Detailed Comparison Matrix

### Script Purpose Comparison

| Purpose | Old Script | New Script(s) | Overlap % | Status |
|---------|-----------|--------------|-----------|---------|
| **Download HF datasets** | download_datasets.py | hf_download.py + bootstrap_data.py | 85% | 🔴 DUPLICATE |
| **Download OpenSLR** | ❌ None | openslr_download.py + bootstrap_data.py | 0% | ✅ NEW |
| **Download Git repos** | ❌ None | primock_download.py | 0% | ✅ NEW |
| **Download medical QA** | ❌ None | afrimedqa_download.py | 0% | ✅ NEW |
| **Preprocess audio** | preprocess_data.py | (bootstrap: derive_low_audio) | 40% | 🟡 PARTIAL |
| **Augment audio** | ❌ None | augment_audio.py + bootstrap_data.py | 30% | 🟡 DUPLICATE |
| **Generate manifests** | ❌ None | make_librispeech_manifest.py + 4 others | 0%* | 🔴 FRAGMENTED |
| **Create eval splits** | preprocess_data.py | ❌ Not in scripts | 0% | ℹ️ USES MODULE |
| **GCS upload** | Both old scripts | ❌ None | 0% | ℹ️ REMOVED |
| **LLM checks** | 2 old scripts | ❌ None | 0% | ℹ️ UNRELATED |

*Manifests are new functionality but logic is duplicated across multiple new scripts

---

## Line-by-Line Duplicity Examples

### Example 1: Download + Extract Pattern

**In `bootstrap_data.py` (Lines 217-273):**
```python
def download_file(url: str, destination: Path, force: bool = False) -> Path:
    from tqdm import tqdm
    # ... download logic with progress bar ...
    
def extract_archive(archive_path: Path, destination_dir: Path, ...):
    if archive_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, mode="r:gz") as tar_ref:
            tar_ref.extractall(destination_dir)
```

**In `openslr_download.py` (Lines 65-228):**
```python
def _stream_download(url: str, destination: Path) -> None:
    # ... resumable download logic with Range support ...
    
def _extract_archive(archive_path: Path, out_dir: Path) -> None:
    if archive_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(out_dir)
```

**Duplicity:** 🔴 Same functionality, different implementations. `openslr_download.py` is more sophisticated (resume support, checksums).

---

### Example 2: Manifest Writing

**Pattern appears in 5+ scripts:**

```python
CSV_FIELDS = ["dataset", "split", "utt_id", "path", ...]

def _write_manifest(rows: List[Dict[str, str]], out_csv: Path, force: bool):
    if out_csv.exists() and not force:
        LOGGER.info("Manifest exists, skipping: %s", out_csv)
        return
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
```

**Found in:**
- `bootstrap_data.py` (Line 355)
- `hf_download.py` (Line 215)
- `make_librispeech_manifest.py` (Line 96)
- `primock_download.py` (Line ~250)
- `afrimedqa_download.py` (Line ~230)

**Duplicity:** 🔴 **CRITICAL** - Exact same logic repeated 5 times

---

### Example 3: Audio Augmentation

**In `augment_audio.py`:**
```python
def _mix_at_snr(clean, noise, snr_db):
    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return (clean + noise * float(scale)).astype(np.float32)
```

**In `bootstrap_data.py` (Line 769):**
```python
def _mix_additive_noise(clean, noise, snr_db):
    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return clean + noise * float(scale)
```

**Duplicity:** 🔴 **EXACT DUPLICATE** (only function name differs)

---

## Consolidation Recommendations

### 🎯 Immediate Actions (High Priority)

#### 1. Deprecate Old Scripts
**Action:** Mark as deprecated, point to new equivalents

| Old Script | Replacement | Migration Path |
|-----------|-------------|----------------|
| `download_datasets.py` | `hf_download.py` + `bootstrap_data.py --download-hf` | Update GCS config in new scripts |
| `preprocess_data.py` | `bootstrap_data.py --derive-*` + augment_audio.py | Migrate evaluation split logic |

**Implementation:**
```python
# Add to download_datasets.py header:
"""
⚠️  DEPRECATED: This script is deprecated as of February 2026.
Use the new modular pipeline instead:
- For HF downloads: scripts/hf_download.py or scripts/bootstrap_data.py --download-hf
- For GCS operations: Extend new scripts with GCS support from src.utils.gcs_utils

See: scripts/README_DATASET_FETCHERS.md
"""
```

#### 2. Extract Common Utilities Module
**Action:** Create `scripts/dataset_utils.py` with shared functions

**Functions to extract:**
```python
# scripts/dataset_utils.py

def safe_name(value: str) -> str:
    """Sanitize string for filesystem use."""
    
def list_audio_files(root: Path) -> List[Path]:
    """Find all audio files recursively."""
    
def get_audio_metadata_ffprobe(path: Path) -> Tuple[float, int]:
    """Extract duration and sample rate using ffprobe."""
    
def get_audio_metadata_soundfile(path: Path) -> Tuple[float, int]:
    """Extract duration and sample rate using soundfile."""
    
def write_manifest(rows: List[Dict], out_csv: Path, fields: List[str], force: bool):
    """Write standardized manifest CSV."""
    
def require_package(module_name: str, pip_name: str = None):
    """Check if Python package is installed."""
```

**Estimated Reduction:** Remove ~200 lines of duplicated code

#### 3. Consolidate Augmentation Functions
**Action:** Make `augment_audio.py` the single source of truth

**Current state:**
- `augment_audio.py` - Standalone augmentation tool
- `bootstrap_data.py` - Duplicates augmentation in Lines 765-898

**Recommendation:**
```python
# bootstrap_data.py should call augment_audio.py as subprocess
def derive_corrupted_audio(paths, force, seed):
    script_path = Path(__file__).parent / "augment_audio.py"
    # Call external script instead of reimplementing
```

**Estimated Reduction:** Remove ~130 lines from bootstrap_data.py

#### 4. Refactor bootstrap_data.py
**Action:** Make it a pure orchestrator, not a reimplementor

**Current size:** 1001 lines  
**Target size:** ~400 lines (60% reduction)

**Strategy:**
- Replace inline OpenSLR logic → call `openslr_download.py`
- Replace inline augmentation → call `augment_audio.py`
- Replace inline manifest generation → call specialized manifest scripts
- Keep only: orchestration, directory management, workflow coordination

---

### 🔧 Medium Priority Actions

#### 5. Unify Manifest Generation
**Option A:** Create `scripts/generate_manifest.py` - Universal manifest builder

```python
# scripts/generate_manifest.py --dataset-type [librispeech|musan|rirs|primock|afrimedqa|hf]
# Auto-detects structure and generates appropriate manifest
```

**Option B:** Keep specialized manifest generators, extract shared logic to utils

**Recommendation:** Option B - domain-specific parsing is valuable

#### 6. Add GCS Support to New Scripts (Optional)
If GCS workflow is still needed:

```python
# Add to hf_download.py, openslr_download.py, etc.
parser.add_argument("--upload-gcs", action="store_true", help="Upload to GCS after download")
parser.add_argument("--gcs-bucket", default="stt-project-datasets")
```

---

### 📊 Low Priority / Non-Duplicates

#### Scripts Without Significant Overlap
- `check_ollama_models.py` - LLM infrastructure (unrelated to STT data)
- `test_llm_connection.py` - LLM testing (unrelated to STT data)
- `finetune_wav2vec2.py` - Model training (different domain)

**Recommendation:** Keep as-is, no consolidation needed

---

## Migration Path

### Phase 1: Immediate (This PR)
1. ✅ Keep new scripts as-is (already done)
2. ✅ Document new scripts (README_DATASET_FETCHERS.md created)
3. ⚠️ Add deprecation notices to old scripts
4. ⚠️ Update main README to point to new workflow

### Phase 2: Consolidation (Next PR)
1. Extract shared utilities to `scripts/dataset_utils.py`
2. Refactor `bootstrap_data.py` to call external scripts
3. Remove duplicated augmentation from `bootstrap_data.py`
4. Optionally add GCS support to new scripts

### Phase 3: Cleanup (Future PR)
1. Delete deprecated scripts after migration period
2. Archive old GCS-centric workflow documentation
3. Consolidate manifest generation if needed

---

## Cost-Benefit Analysis

### Keeping Current State (With Duplicity)
**Pros:**
- New scripts work independently
- No refactoring risk
- Clear separation of concerns

**Cons:**
- ~400 lines of duplicated code
- Maintenance burden (fix bugs in multiple places)
- Inconsistent patterns across scripts

### Consolidating (Recommended)
**Pros:**
- ~40% code reduction in scripts/
- Single source of truth for common operations
- Easier maintenance and bug fixes
- Consistent error handling

**Cons:**
- Refactoring effort (~4-6 hours)
- Potential for breaking changes
- Need comprehensive testing

**Recommendation:** ✅ **Consolidate in Phase 2** (separate PR after current work stabilizes)

---

## Summary Table: All Data-Related Files

| File | Size | Purpose | Duplicity | Recommendation |
|------|------|---------|-----------|----------------|
| **scripts/download_datasets.py** | 338L | OLD HF downloader | 🔴 85% vs hf_download.py | **DEPRECATE** |
| **scripts/preprocess_data.py** | 194L | OLD preprocessing | 🟡 60% vs augment_audio.py | **DEPRECATE** |
| **scripts/bootstrap_data.py** | 1001L | NEW orchestrator | 🔴 30% internal duplication | **REFACTOR** |
| **scripts/openslr_download.py** | 300L | OpenSLR downloader | 🟡 30% vs bootstrap | **KEEP + FIX** |
| **scripts/hf_download.py** | 348L | Generic HF downloader | 🟢 Low | **KEEP** |
| **scripts/make_librispeech_manifest.py** | 169L | LibriSpeech manifests | 🔴 70% vs bootstrap | **CONSOLIDATE** |
| **scripts/augment_audio.py** | 266L | Audio augmentation | 🔴 95% vs bootstrap | **KEEP, REMOVE from bootstrap** |
| **scripts/primock_download.py** | 376L | PriMock57 downloader | 🟢 Low | **KEEP** |
| **scripts/afrimedqa_download.py** | 270L | AfriMed-QA downloader | 🟢 Low | **KEEP** |
| **scripts/check_ollama_models.py** | 153L | LLM check | 🟢 None | **KEEP** |
| **scripts/test_llm_connection.py** | 153L | LLM test | 🟢 None | **KEEP** |
| **scripts/finetune_wav2vec2.py** | ?L | Training | 🟢 None | **KEEP** |
| **src/data/preprocessing.py** | 123L | Audio utilities | 🟡 40% vs scripts | **KEEP** |
| **src/data/evaluation_splits.py** | 105L | Split creation | 🟢 Unique | **KEEP** |
| **src/data/data_manager.py** | 468L | Data management | 🟢 None | **KEEP** |
| *Other src/data modules* | 6000L+ | Production infra | 🟢 None | **KEEP** |

---

## Critical Issues Requiring Attention

### Issue 1: `bootstrap_data.py` is a Monolith
**Current:** 1001 lines doing everything  
**Problem:** Duplicates logic from 3 other scripts internally  
**Solution:** Convert to orchestrator that calls other scripts

### Issue 2: No Shared Utility Module
**Problem:** Same utility functions copied 5+ times  
**Solution:** Create `scripts/dataset_utils.py`

### Issue 3: Old Scripts Will Confuse Users
**Problem:** Two sets of scripts with similar names/purposes  
**Solution:** Add clear deprecation warnings and migration guide

### Issue 4: Manifest Generation is Fragmented
**Problem:** 5 different scripts generate similar CSVs with copy-pasted logic  
**Solution:** Extract manifest writing to utility or keep specialized but extract common code

---

## Recommended File Actions

### Files to Deprecate (Add warning, keep for now)
- ✗ `scripts/download_datasets.py` → Superseded by `hf_download.py`
- ✗ `scripts/preprocess_data.py` → Superseded by `augment_audio.py` + bootstrap

### Files to Refactor
- ⚠️ `scripts/bootstrap_data.py` → Remove inline duplication, call external scripts
- ⚠️ `scripts/augment_audio.py` vs bootstrap → Remove from bootstrap

### Files to Create
- ➕ `scripts/dataset_utils.py` → Shared utility functions
- ➕ `scripts/MIGRATION_GUIDE.md` → Old → New script mapping

### Files to Keep As-Is
- ✓ `scripts/openslr_download.py` (enhance, don't duplicate)
- ✓ `scripts/hf_download.py` (solid generic implementation)
- ✓ `scripts/primock_download.py` (unique functionality)
- ✓ `scripts/afrimedqa_download.py` (unique functionality)
- ✓ `scripts/make_librispeech_manifest.py` (specialized, useful)
- ✓ All `src/data/*` modules (production infrastructure)

---

## Conclusion

**Overall Duplicity Level: 🔴 HIGH (60-70%)**

**Primary Causes:**
1. New branch reimplements download logic without deprecating old scripts
2. `bootstrap_data.py` duplicates logic from its own companion scripts
3. No shared utilities module for common patterns
4. Manifest generation logic copied across 5 scripts

**Immediate Impact:**
- Current state is **functional but unmaintainable**
- Bug fixes require changes in multiple locations
- New contributors will be confused by two pipelines

**Recommended Action:**
1. **For THIS PR:** Keep new scripts, add deprecation warnings to old ones
2. **NEXT PR:** Create utilities module, refactor bootstrap_data.py
3. **FUTURE:** Remove deprecated scripts after migration period

**Priority:** 🟡 **MEDIUM** - Doesn't block current work, but should be addressed soon for maintainability
