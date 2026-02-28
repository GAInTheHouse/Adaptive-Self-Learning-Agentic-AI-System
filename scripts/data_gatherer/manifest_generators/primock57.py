#!/usr/bin/env python3
"""
PriMock57 manifest generator.

Generates manifests for PriMock57 medical consultation dataset,
parsing audio files, transcripts, and consultation notes.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_utils import (
    CSV_FIELDS,
    get_audio_metadata_ffprobe,
    write_manifest,
)


LOGGER = logging.getLogger("primock57_generator")


def parse_transcript(transcript_path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse PriMock57 transcript file.
    
    Args:
        transcript_path: Path to transcript text file
        
    Returns:
        List of (speaker, text, timestamp) tuples
    """
    utterances = []
    
    if not transcript_path.exists():
        return utterances
    
    try:
        with transcript_path.open("r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract speaker and text (format: "Speaker: text")
            match = re.match(r'^([^:]+):\s*(.+)$', line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                utterances.append((speaker, text, ""))
            else:
                utterances.append(("unknown", line, ""))
        
        # If no structured format, return full text as single utterance
        if not utterances and content.strip():
            utterances.append(("unknown", content.strip(), ""))
            
    except Exception as exc:
        LOGGER.warning("Failed to parse transcript %s: %s", transcript_path.name, exc)
    
    return utterances


def generate(data_dir: Path, manifest_dir: Path, force: bool) -> List[Path]:
    """
    Generate PriMock57 manifest CSV file.
    
    Args:
        data_dir: Root directory containing audio/, transcripts/, notes/ subdirs
        manifest_dir: Output directory for manifest CSV files
        force: If True, overwrite existing manifests
        
    Returns:
        List containing single manifest file path
    """
    from tqdm import tqdm
    
    audio_dir = data_dir / "audio"
    transcripts_dir = data_dir / "transcripts"
    notes_dir = data_dir / "notes"
    
    if not audio_dir.exists():
        LOGGER.warning("Audio directory not found: %s", audio_dir)
        return []
    
    # Find all audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    
    if not audio_files:
        LOGGER.warning("No WAV files found in: %s", audio_dir)
        return []
    
    LOGGER.info("Found %d PriMock57 audio files", len(audio_files))
    
    manifest_rows = []
    
    for audio_path in tqdm(audio_files, desc="primock57"):
        consultation_id = audio_path.stem
        
        # Get audio metadata using ffprobe (WAV files)
        try:
            duration, sample_rate = get_audio_metadata_ffprobe(audio_path)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", audio_path.name, exc)
            continue
        
        # Find corresponding transcript
        transcript_path = transcripts_dir / f"{consultation_id}.txt"
        if not transcript_path.exists():
            # Try alternative patterns
            for pattern in ["*.txt", "*.trans.txt"]:
                matches = list(transcripts_dir.glob(pattern))
                for match in matches:
                    if consultation_id in match.stem:
                        transcript_path = match
                        break
        
        # Parse transcript
        utterances = parse_transcript(transcript_path)
        full_text = " ".join([utt[1] for utt in utterances])
        
        # Extract speakers
        speakers = list(set([utt[0] for utt in utterances if utt[0] != "unknown"]))
        speaker_str = ",".join(speakers) if speakers else ""
        
        # Create manifest entry
        manifest_rows.append({
            "dataset": "primock57",
            "split": "full",
            "utt_id": consultation_id,
            "path": str(audio_path.absolute()),
            "duration_seconds": f"{duration:.6f}",
            "sampling_rate": str(sample_rate),
            "text": full_text,
            "speaker": speaker_str,
            "accent": "uk",
        })
    
    # Write manifest
    manifest_path = manifest_dir / "primock57__full.csv"
    write_manifest(manifest_rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
    
    return [manifest_path]
