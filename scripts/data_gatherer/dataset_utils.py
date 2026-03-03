#!/usr/bin/env python3
"""
Shared utilities for data gathering scripts.

This module consolidates common functions used across all data source plugins
to eliminate code duplication and ensure consistent behavior.
"""

from __future__ import annotations

import csv
import importlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def configure_logging(name: str = "data_gatherer") -> logging.Logger:
    """
    Configure standard logging format for all data gathering scripts.
    
    Args:
        name: Logger name to use
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler - ERROR level only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def safe_name(value: str) -> str:
    """
    Sanitize string to create safe filenames and identifiers.
    
    Replaces non-alphanumeric characters (except ._-) with underscores.
    
    Args:
        value: String to sanitize
        
    Returns:
        Sanitized string safe for use in filenames
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def require_package(module_name: str, pip_name: Optional[str] = None) -> None:
    """
    Check if a Python package is installed, raise error if missing.
    
    Args:
        module_name: Name of module to import (e.g., 'datasets')
        pip_name: Name to use in pip install command (if different from module_name)
        
    Raises:
        RuntimeError: If package is not installed
    """
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        install_name = pip_name or module_name
        raise RuntimeError(
            f"Missing Python package '{install_name}'. Install with: pip install {install_name}"
        ) from exc


def list_audio_files(
    directory: Path, 
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Recursively find all audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of extensions to match (e.g., ['.wav', '.flac']).
                   If None, uses default: .wav, .flac, .mp3, .ogg, .m4a
        
    Returns:
        Sorted list of audio file paths
    """
    if extensions is None:
        extensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory.rglob(f"*{ext}"))
    
    return sorted(audio_files)


def get_audio_metadata_soundfile(audio_path: Path) -> Tuple[float, int]:
    """
    Extract audio duration and sample rate using soundfile library.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (duration_seconds, sample_rate)
        
    Raises:
        RuntimeError: If soundfile is not installed or file cannot be read
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Missing soundfile package. Install with: pip install soundfile"
        ) from exc
    
    info = sf.info(str(audio_path))
    return float(info.duration), int(info.samplerate)


def get_audio_metadata_ffprobe(audio_path: Path) -> Tuple[float, int]:
    """
    Extract audio duration and sample rate using ffprobe.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (duration_seconds, sample_rate)
        
    Raises:
        RuntimeError: If ffprobe is not available or execution fails
    """
    import json
    import shutil
    import subprocess
    
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe not found. Install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg"
        )
    
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


def write_manifest(
    rows: List[Dict[str, str]], 
    output_path: Path, 
    fieldnames: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    Write manifest CSV file with standardized format.
    
    Args:
        rows: List of dictionaries containing manifest data
        output_path: Path to output CSV file
        fieldnames: List of CSV column names (uses CSV_FIELDS if None)
        force: If True, overwrite existing file
        
    Raises:
        FileExistsError: If file exists and force=False
    """
    if output_path.exists() and not force:
        logging.getLogger("dataset_utils").info(
            "Manifest exists, skipping: %s", output_path
        )
        return
    
    if fieldnames is None:
        fieldnames = CSV_FIELDS
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    logging.getLogger("dataset_utils").info(
        "Wrote manifest: %s (%d rows)", output_path, len(rows)
    )


def pick_field(record: Dict[str, object], candidates: List[str], default: str = "") -> str:
    """
    Extract first available field from a record, trying multiple field names.
    
    Args:
        record: Dictionary to search
        candidates: List of field names to try in order
        default: Default value if no field is found
        
    Returns:
        Value of first matching field, or default if none found
    """
    for key in candidates:
        value = record.get(key)
        if value is not None:
            result = str(value).strip()
            if result:
                return result
    return default
