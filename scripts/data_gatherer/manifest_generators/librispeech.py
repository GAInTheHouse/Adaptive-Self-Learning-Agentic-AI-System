#!/usr/bin/env python3
"""
LibriSpeech manifest generator.

Extracts from make_librispeech_manifest.py logic for parsing
OpenSLR SLR12 LibriSpeech directory structure with .trans.txt transcripts.
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
    safe_name,
    write_manifest,
)


LOGGER = logging.getLogger("librispeech_generator")


def load_transcript_map(split_dir: Path) -> Dict[str, str]:
    """
    Load LibriSpeech transcript mappings from .trans.txt files.
    
    Args:
        split_dir: Directory containing speaker subdirectories with .trans.txt files
        
    Returns:
        Dictionary mapping utterance_id -> transcript text
    """
    mapping: Dict[str, str] = {}
    
    for trans_path in split_dir.rglob("*.trans.txt"):
        with trans_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                utt_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                mapping[utt_id] = text
    
    return mapping


def build_rows_for_split(split_dir: Path, split_name: str) -> List[Dict[str, str]]:
    """
    Build manifest rows for a LibriSpeech split directory.
    
    Args:
        split_dir: Path to split directory (e.g., LibriSpeech/dev-clean)
        split_name: Name of split (e.g., 'dev-clean')
        
    Returns:
        List of manifest row dictionaries
    """
    transcript_map = load_transcript_map(split_dir)
    rows: List[Dict[str, str]] = []
    
    for flac_path in sorted(split_dir.rglob("*.flac")):
        utt_id = flac_path.stem
        
        try:
            duration, sampling_rate = get_audio_metadata_soundfile(flac_path)
        except Exception as exc:
            LOGGER.warning("Failed to get metadata for %s: %s", flac_path, exc)
            continue
        
        # Extract speaker ID from utterance ID (format: speakerID-chapterID-uttID)
        speaker = utt_id.split("-")[0] if "-" in utt_id else ""
        
        rows.append({
            "dataset": "librispeech",
            "split": split_name,
            "utt_id": utt_id,
            "path": str(flac_path.absolute()),
            "duration_seconds": f"{duration:.6f}",
            "sampling_rate": str(sampling_rate),
            "text": transcript_map.get(utt_id, ""),
            "speaker": speaker,
            "accent": "",
        })
    
    return rows


def _is_single_split(candidate_dir: Path) -> bool:
    """
    Return True if candidate_dir is already a single LibriSpeech split directory.

    A split directory contains numeric speaker-ID subdirectories (e.g. 1272/, 1673/)
    rather than named split subdirectories (e.g. dev-clean/, dev-other/).
    """
    subdirs = [p for p in candidate_dir.iterdir() if p.is_dir()]
    return bool(subdirs) and all(d.name.isdigit() for d in subdirs)


def generate(data_dir: Path, manifest_dir: Path, force: bool) -> List[Path]:
    """
    Generate LibriSpeech manifest CSV files.

    Handles two layouts:

    * Multi-split root (HF / full download)::

        data_dir/LibriSpeech/dev-clean/...
        data_dir/LibriSpeech/dev-other/...

    * Single-split root (OpenSLR per-split download, e.g. dev-clean.tar.gz)::

        data_dir/LibriSpeech/dev-clean/<speaker_id>/...
        -- or --
        data_dir/<speaker_id>/...   (extract_path points directly at the split)

    Args:
        data_dir: Root directory for downloaded data
        manifest_dir: Output directory for manifest CSV files
        force: If True, overwrite existing manifests

    Returns:
        List of generated manifest file paths
    """
    manifest_paths = []

    # Prefer data_dir/LibriSpeech if it exists, otherwise treat data_dir as root
    librispeech_root = data_dir / "LibriSpeech"
    if not librispeech_root.exists():
        librispeech_root = data_dir

    if not librispeech_root.exists():
        LOGGER.warning("LibriSpeech root not found: %s", librispeech_root)
        return manifest_paths

    split_dirs = [p for p in librispeech_root.iterdir() if p.is_dir()]

    if not split_dirs:
        LOGGER.warning("No LibriSpeech split directories found in: %s", librispeech_root)
        return manifest_paths

    # Detect whether librispeech_root is itself a single split (speaker-ID subdirs)
    # This happens when extract_path already points at e.g. LibriSpeech/dev-clean.
    if _is_single_split(librispeech_root):
        split_name = librispeech_root.name
        LOGGER.info("Detected single-split layout; treating %s as split '%s'", librispeech_root, split_name)
        rows = build_rows_for_split(librispeech_root, split_name)
        if not rows:
            LOGGER.warning("No audio files found in: %s", librispeech_root)
            return manifest_paths
        manifest_path = manifest_dir / f"librispeech__{split_name}.csv"
        write_manifest(rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
        manifest_paths.append(manifest_path)
        return manifest_paths

    # Multi-split layout: each subdirectory is a named split
    for split_dir in sorted(split_dirs):
        split_name = split_dir.name
        LOGGER.info("Generating manifest for LibriSpeech/%s", split_name)

        rows = build_rows_for_split(split_dir, split_name)

        if not rows:
            LOGGER.warning("No audio files found in: %s", split_dir)
            continue

        manifest_path = manifest_dir / f"librispeech__{split_name}.csv"
        write_manifest(rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
        manifest_paths.append(manifest_path)

    return manifest_paths
