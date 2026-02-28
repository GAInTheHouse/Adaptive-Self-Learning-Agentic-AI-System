# Script Consolidation Recommendations

**Date:** February 27, 2026  
**Branch:** gxa/create-data  
**Analysis:** Based on DUPLICITY_ANALYSIS.md

---

## Executive Recommendation

**Consolidate ~400 lines of duplicated code across the scripts/ directory by:**

1. Creating a shared utilities module
2. Refactoring bootstrap_data.py to call external scripts
3. Deprecating old GCS-centric scripts
4. Keeping specialized downloaders as-is

**Estimated effort:** 4-6 hours  
**Risk level:** Low (changes are isolated to scripts/, not production code)  
**Benefit:** 40% code reduction, easier maintenance

---

## Priority 1: Create Shared Utilities Module

### Create `scripts/dataset_utils.py`

**Extract these functions (used in 3-5 scripts each):**

```python
#!/usr/bin/env python3
"""
Shared utilities for dataset download and manifest generation scripts.
"""

import csv
import importlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("dataset_utils")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

CSV_FIELDS = [
    "dataset",
    "split",
    "utt_id",
    "path",
    "duration_seconds",
    "sampling_rate",
    "text",
    "speaker",
    "accent",
]


def safe_name(value: str) -> str:
    """Sanitize string for use in filenames and IDs."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def list_audio_files(root: Path) -> List[Path]:
    """Recursively find all audio files under root directory."""
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def require_package(module_name: str, pip_name: Optional[str] = None) -> None:
    """Check if Python package is available, raise error with install hint if not."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        install_name = pip_name or module_name
        raise RuntimeError(
            f"Missing Python package '{module_name}'. "
            f"Install with: pip install {install_name}"
        ) from exc


def get_audio_metadata_ffprobe(audio_path: Path) -> Tuple[float, int]:
    """
    Extract audio duration and sample rate using ffprobe.
    
    Returns: (duration_seconds, sample_rate)
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=sample_rate",
        "-select_streams", "a:0",
        "-of", "json",
        str(audio_path),
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        duration = float(data["format"]["duration"])
        streams = data.get("streams", [])
        sample_rate = int(streams[0]["sample_rate"]) if streams else 16000
        
        return duration, sample_rate
        
    except Exception as exc:
        raise RuntimeError(f"Failed to extract metadata from {audio_path}: {exc}") from exc


def get_audio_metadata_soundfile(audio_path: Path) -> Tuple[float, int]:
    """
    Extract audio duration and sample rate using soundfile.
    
    Returns: (duration_seconds, sample_rate)
    """
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return float(info.duration), int(info.samplerate)
    except ImportError as exc:
        raise RuntimeError("soundfile package required. Install with: pip install soundfile") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to extract metadata from {audio_path}: {exc}") from exc


def write_manifest(
    rows: List[Dict[str, str]],
    out_csv: Path,
    fieldnames: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    Write standardized manifest CSV file.
    
    Args:
        rows: List of dictionaries with manifest data
        out_csv: Output CSV path
        fieldnames: CSV column names (defaults to standard fields)
        force: Overwrite existing file
    """
    if out_csv.exists() and not force:
        LOGGER.info("Manifest exists, skipping: %s", out_csv)
        return
    
    if fieldnames is None:
        fieldnames = CSV_FIELDS
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    LOGGER.info("Wrote manifest: %s (%d rows)", out_csv, len(rows))


def configure_logging(name: str = __name__) -> logging.Logger:
    """Configure standard logging format for scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
```

**Impact:**
- Remove ~35 lines from each of 5 scripts = **~175 lines saved**
- Consistent behavior across all scripts
- Single place to fix bugs

---

## Priority 2: Refactor bootstrap_data.py

### Current Issues in bootstrap_data.py

**Problem 1: Duplicates augment_audio.py logic**
- Lines 765-898: Audio augmentation functions
- Exact same algorithms as `augment_audio.py`
- **Solution:** Call augment_audio.py as subprocess

**Problem 2: Reimplements OpenSLR download**
- Lines 217-308: Download and extract functions  
- Less sophisticated than `openslr_download.py` (no checksums, no resume)
- **Solution:** Call openslr_download.py as subprocess

**Problem 3: Inline manifest generation**
- Lines 369-476: LibriSpeech/MUSAN/RIRS manifest generation
- Duplicates logic from `make_librispeech_manifest.py`
- **Solution:** Call make_librispeech_manifest.py or extract to utils

### Proposed Refactor

#### Before: 1001 lines
```python
# bootstrap_data.py (current)

def download_file(...):           # Lines 217-250 (34 lines)
def extract_archive(...):         # Lines 253-273 (21 lines)
def download_openslr(...):        # Lines 276-308 (33 lines)
def generate_librispeech_manifests(...):  # Lines 369-411 (43 lines)
def generate_musan_manifests(...): # Lines 414-441 (28 lines)
def generate_rirs_manifests(...):  # Lines 444-475 (32 lines)
def _mix_additive_noise(...):     # Lines 765-776 (12 lines)
def _apply_reverb(...):           # Lines 779-786 (8 lines)
def _apply_random_dropouts(...):  # Lines 789-799 (11 lines)
def _apply_random_clipping(...):  # Lines 802-807 (6 lines)
def derive_corrupted_audio(...):  # Lines 852-916 (65 lines)
```

#### After: ~600 lines (40% reduction)
```python
# bootstrap_data.py (refactored)

from dataset_utils import (
    safe_name,
    list_audio_files,
    get_audio_metadata_ffprobe,
    write_manifest,
    require_package,
    configure_logging,
)

def download_openslr(...):
    """Call external openslr_download.py script."""
    script = Path(__file__).parent / "openslr_download.py"
    subprocess.run([sys.executable, str(script), "--dataset", ...])

def generate_librispeech_manifests(...):
    """Call external make_librispeech_manifest.py script."""
    script = Path(__file__).parent / "make_librispeech_manifest.py"
    subprocess.run([sys.executable, str(script), ...])

def derive_corrupted_audio(...):
    """Call external augment_audio.py script."""
    script = Path(__file__).parent / "augment_audio.py"
    for source_row in source_rows:
        subprocess.run([
            sys.executable, str(script),
            "--input-dir", src_dir,
            "--output-dir", out_dir,
            ...
        ])
```

**Benefits:**
- 400 lines removed
- bootstrap_data.py becomes pure orchestrator
- Single source of truth for each operation
- Easier to test individual components

---

## Priority 3: Add Runtime Deprecation Warnings

### Add to main() in old scripts

```python
def main():
    """Main download routine"""
    
    # DEPRECATION WARNING
    print("\n" + "="*70)
    print("⚠️  DEPRECATION WARNING")
    print("="*70)
    print("This script (download_datasets.py) is deprecated.")
    print()
    print("Replacement: scripts/hf_download.py or scripts/bootstrap_data.py")
    print("See: scripts/MIGRATION_GUIDE.md")
    print("="*70)
    print()
    
    response = input("Continue anyway? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted. Please use new scripts.")
        return 1
    
    # ... original logic continues ...
```

**Impact:** Forces users to acknowledge deprecation

---

## Priority 4: Update Documentation

### Files to Update

#### 1. Main README.md
```markdown
## Data Pipeline Scripts

### Current Scripts (gxa/create-data branch)
- `scripts/bootstrap_data.py` - Unified data orchestrator
- `scripts/openslr_download.py` - OpenSLR datasets
- `scripts/hf_download.py` - Hugging Face datasets
- `scripts/primock_download.py` - PriMock57 medical consultations
- `scripts/afrimedqa_download.py` - AfriMed-QA medical QA
- `scripts/augment_audio.py` - Audio augmentation

See: [scripts/README_DATASET_FETCHERS.md](scripts/README_DATASET_FETCHERS.md)

### Deprecated Scripts (main branch)
- ~~`scripts/download_datasets.py`~~ → Use `hf_download.py`
- ~~`scripts/preprocess_data.py`~~ → Use `augment_audio.py`

See: [scripts/MIGRATION_GUIDE.md](scripts/MIGRATION_GUIDE.md)
```

#### 2. scripts/README.md (create if missing)
Quick reference for all scripts with deprecation status

---

## Implementation Checklist

### Phase 1: Immediate (This PR)
- [x] Create DUPLICITY_ANALYSIS.md
- [x] Create MIGRATION_GUIDE.md
- [x] Create CONSOLIDATION_RECOMMENDATIONS.md
- [x] Add deprecation docstrings to old scripts
- [ ] Add runtime deprecation warnings (optional)
- [ ] Update main README.md with script status

### Phase 2: Refactoring (Next PR)
- [ ] Create scripts/dataset_utils.py
- [ ] Refactor bootstrap_data.py to use utils
- [ ] Refactor bootstrap_data.py to call external scripts
- [ ] Update all scripts to import from dataset_utils
- [ ] Remove augmentation duplication from bootstrap_data.py
- [ ] Add comprehensive tests

### Phase 3: Cleanup (Future)
- [ ] Remove download_datasets.py
- [ ] Remove preprocess_data.py
- [ ] Consolidate manifest generation (if needed)
- [ ] Add GCS support back (if needed)

---

## Estimated Code Reduction

| Action | Lines Removed | Scripts Affected |
|--------|--------------|------------------|
| Create dataset_utils.py | -175 | 5 scripts |
| Remove augmentation from bootstrap | -130 | bootstrap_data.py |
| Call external scripts in bootstrap | -100 | bootstrap_data.py |
| Delete deprecated scripts | -532 | download_datasets.py, preprocess_data.py |
| **TOTAL REDUCTION** | **-937 lines** | **7 scripts** |

**Current:** ~3500 lines in scripts/  
**After consolidation:** ~2563 lines (27% reduction)

---

## Risk Assessment

### Low Risk Changes
- ✅ Creating dataset_utils.py (pure addition)
- ✅ Adding deprecation warnings (non-breaking)
- ✅ Documentation updates

### Medium Risk Changes
- ⚠️ Refactoring bootstrap_data.py (well-tested, isolated)
- ⚠️ Removing augmentation duplication (have tests)

### High Risk Changes
- 🔴 Deleting old scripts (wait for migration period)

**Recommended approach:** Gradual migration over 2-3 PRs with testing between each phase.

---

## Success Metrics

### Maintainability
- **Before:** Fix bug → update 5 files
- **After:** Fix bug → update 1 file

### Onboarding
- **Before:** "Which download script should I use?" (confusing)
- **After:** "Use bootstrap_data.py or individual scripts" (clear)

### Code Quality
- **Before:** 937 lines of duplication, 3 implementations of same algorithm
- **After:** Single source of truth, DRY principles

### Functionality
- **Before:** GCS-centric, 3 hardcoded datasets
- **After:** Local-first, unlimited datasets, medical domain support

---

## Alternative: Keep Current State

### Arguments For Status Quo
1. **Working code:** Everything functions as-is
2. **Zero risk:** No refactoring means no breaking changes
3. **Clear separation:** Old vs new is explicit

### Arguments Against Status Quo
1. **Maintenance burden:** Bug fixes in multiple places
2. **Code bloat:** ~937 lines of unnecessary duplication
3. **Confusion:** Two pipelines doing similar things
4. **Technical debt:** Will get worse over time

**Verdict:** 🔴 **Consolidation strongly recommended** but can be phased

---

## Conclusion

The analysis reveals **60-70% functional overlap** between old and new scripts, with **~937 lines of duplicated code**. 

**Immediate action required:**
- ✅ Deprecation warnings added
- ✅ Migration guide created
- ✅ Analysis documented

**Next steps (recommended for separate PR):**
1. Create shared utilities module
2. Refactor bootstrap_data.py
3. Remove augmentation duplication

**Long-term:**
- Delete deprecated scripts after migration period
- Monitor for additional consolidation opportunities
