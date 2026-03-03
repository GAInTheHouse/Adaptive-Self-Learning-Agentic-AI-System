#!/usr/bin/env python3
"""
RIRS_NOISES manifest generator.

Generates manifests for Room Impulse Response and Noise Database,
grouping files by subdirectory (pointsource_noises, real_rirs_isotropic_noises, simulated_rirs).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_utils import (
    CSV_FIELDS,
    get_audio_metadata_soundfile,
    list_audio_files,
    write_manifest,
)


LOGGER = logging.getLogger("rirs_generator")


def generate(data_dir: Path, manifest_dir: Path, force: bool) -> List[Path]:
    """
    Generate RIRS_NOISES manifest CSV files, grouped by subdirectory.
    
    Args:
        data_dir: Root directory containing RIRS_NOISES subdirectory
        manifest_dir: Output directory for manifest CSV files
        force: If True, overwrite existing manifests
        
    Returns:
        List of generated manifest file paths
    """
    from tqdm import tqdm
    
    manifest_paths = []
    
    # Find RIRS root
    rirs_root = data_dir / "RIRS_NOISES"
    if not rirs_root.exists():
        rirs_root = data_dir
    
    if not rirs_root.exists():
        LOGGER.warning("RIRS_NOISES directory not found: %s", data_dir)
        return manifest_paths
    
    # Find all audio files recursively
    audio_files = list_audio_files(rirs_root)
    
    if not audio_files:
        LOGGER.warning("No audio files found in: %s", rirs_root)
        return manifest_paths
    
    LOGGER.info("Found %d RIRS audio files", len(audio_files))
    
    # Build rows with split based on subdirectory
    rows: List[Dict[str, str]] = []
    
    for idx, audio_path in enumerate(tqdm(audio_files, desc="rirs_noises"), start=1):
        try:
            duration, sample_rate = get_audio_metadata_soundfile(audio_path)
        except Exception as exc:
            LOGGER.warning("Failed to process %s: %s", audio_path, exc)
            continue
        
        # Determine split from relative path structure
        try:
            relative_parts = audio_path.relative_to(rirs_root).parts
            split = relative_parts[0] if relative_parts else "default"
        except ValueError:
            split = "default"
        
        rows.append({
            "dataset": "rirs_noises",
            "split": split,
            "utt_id": f"rirs_{idx:08d}",
            "path": str(audio_path.absolute()),
            "duration_seconds": f"{duration:.6f}",
            "sampling_rate": str(sample_rate),
            "text": "",
            "speaker": "",
            "accent": "",
        })
    
    # Group by split and write separate manifests
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        split = row["split"]
        grouped.setdefault(split, []).append(row)
    
    for split, split_rows in grouped.items():
        manifest_path = manifest_dir / f"rirs_noises__{split}.csv"
        write_manifest(split_rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
        manifest_paths.append(manifest_path)
    
    return manifest_paths
