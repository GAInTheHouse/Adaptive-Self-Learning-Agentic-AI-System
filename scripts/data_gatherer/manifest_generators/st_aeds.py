#!/usr/bin/env python3
"""
ST-AEDS manifest generator.

Generates manifests for ST-AEDS-20180100 (Surfingtech American English Dataset).
The dataset has a simple structure:
- Audio files: f0001_us_f0001_00001.wav (speaker_country_speaker_utterance pattern)
- Transcript file: text.txt with tab-separated format: filename.wav<tab>transcript text
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


LOGGER = logging.getLogger("st_aeds_generator")


def load_transcript_map(text_file: Path) -> Dict[str, str]:
    """
    Load ST-AEDS transcript mappings from text.txt file.
    
    Format: filename.wav<tab>transcript text
    
    Args:
        text_file: Path to text.txt file
        
    Returns:
        Dictionary mapping filename (without .wav) -> transcript text
    """
    mapping: Dict[str, str] = {}
    
    if not text_file.exists():
        LOGGER.warning("Transcript file not found: %s", text_file)
        return mapping
    
    with text_file.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            
            # Split by tab
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                LOGGER.warning("Invalid format at line %d: %s", line_num, line[:50])
                continue
            
            filename, text = parts
            # Remove .wav extension from filename for matching
            utt_id = filename.replace(".wav", "")
            mapping[utt_id] = text.strip()
    
    LOGGER.info("Loaded %d transcripts from %s", len(mapping), text_file.name)
    return mapping


def extract_speaker_from_filename(filename: str) -> str:
    """
    Extract speaker ID from ST-AEDS filename pattern.
    
    Pattern: f0001_us_f0001_00001.wav
    Format: speaker_country_speaker_utterance
    
    Args:
        filename: Audio filename (with or without .wav extension)
        
    Returns:
        Speaker ID (e.g., "f0001")
    """
    # Remove extension if present
    name = filename.replace(".wav", "")
    
    # Split by underscore and take first part
    parts = name.split("_")
    if len(parts) >= 1:
        return parts[0]
    
    return ""


def generate(data_dir: Path, manifest_dir: Path, force: bool) -> List[Path]:
    """
    Generate ST-AEDS manifest CSV file.
    
    Args:
        data_dir: Root directory containing ST-AEDS audio files and text.txt
        manifest_dir: Output directory for manifest CSV files
        force: If True, overwrite existing manifests
        
    Returns:
        List of generated manifest file paths
    """
    from tqdm import tqdm
    
    manifest_paths = []
    
    # Load transcript mappings - check multiple possible locations
    # The archive might extract to a subdirectory or directly to the parent
    text_file = data_dir / "text.txt"
    
    if not text_file.exists():
        # Check parent directory (in case extract_path was set but archive extracted directly)
        parent_dir = data_dir.parent
        parent_text_file = parent_dir / "text.txt"
        if parent_text_file.exists():
            LOGGER.info("Found text.txt in parent directory: %s", parent_dir)
            data_dir = parent_dir
            text_file = parent_text_file
        else:
            # Check for ST-AEDS extracted subdirectory
            st_aeds_subdir = data_dir / "ST-AEDS-20180100_1-OS"
            if st_aeds_subdir.exists():
                LOGGER.info("Using extracted subdirectory: %s", st_aeds_subdir)
                data_dir = st_aeds_subdir
                text_file = st_aeds_subdir / "text.txt"
    
    transcript_map = load_transcript_map(text_file)
    
    if not transcript_map:
        LOGGER.error("No transcripts found in: %s", text_file)
        return manifest_paths
    
    # Find all WAV files
    audio_files = list_audio_files(data_dir, extensions=[".wav"])
    
    if not audio_files:
        LOGGER.warning("No audio files found in: %s", data_dir)
        return manifest_paths
    
    LOGGER.info("Found %d audio files", len(audio_files))
    
    # Generate manifest
    rows: List[Dict[str, str]] = []
    
    for audio_path in tqdm(sorted(audio_files), desc="st_aeds"):
        # Extract utterance ID (filename without extension)
        utt_id = audio_path.stem
        
        # Skip if no transcript
        if utt_id not in transcript_map:
            LOGGER.warning("No transcript for: %s", utt_id)
            continue
        
        # Get audio metadata
        try:
            duration, sample_rate = get_audio_metadata_soundfile(audio_path)
        except Exception as exc:
            LOGGER.warning("Failed to get metadata for %s: %s", audio_path, exc)
            continue
        
        # Extract speaker ID
        speaker = extract_speaker_from_filename(utt_id)
        
        rows.append({
            "dataset": "st_aeds",
            "split": "train",  # ST-AEDS doesn't have predefined splits
            "utt_id": utt_id,
            "path": str(audio_path.absolute()),
            "duration_seconds": f"{duration:.6f}",
            "sampling_rate": str(sample_rate),
            "text": transcript_map[utt_id],
            "speaker": speaker,
            "accent": "us",  # American English dataset
        })
    
    if not rows:
        LOGGER.warning("No valid audio-transcript pairs found")
        return manifest_paths
    
    # Write manifest
    manifest_path = manifest_dir / "st_aeds__train.csv"
    write_manifest(rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
    manifest_paths.append(manifest_path)
    
    LOGGER.info("Generated manifest with %d utterances: %s", len(rows), manifest_path.name)
    
    return manifest_paths
