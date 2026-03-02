#!/usr/bin/env python3
"""
MUSAN noise corpus manifest generator.

Generates manifests for MUSAN categories: music, speech, noise.
Each category becomes a separate CSV.
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


LOGGER = logging.getLogger("musan_generator")


def generate(data_dir: Path, manifest_dir: Path, force: bool) -> List[Path]:
    """
    Generate MUSAN manifest CSV files, one per category.
    
    Args:
        data_dir: Root directory containing musan/ subdirectory
        manifest_dir: Output directory for manifest CSV files
        force: If True, overwrite existing manifests
        
    Returns:
        List of generated manifest file paths
    """
    from tqdm import tqdm
    
    manifest_paths = []
    
    # Find MUSAN root
    musan_root = data_dir / "musan"
    if not musan_root.exists():
        musan_root = data_dir
    
    if not musan_root.exists():
        LOGGER.warning("MUSAN directory not found: %s", data_dir)
        return manifest_paths
    
    # Process each category directory (music, speech, noise)
    categories = [p for p in musan_root.iterdir() if p.is_dir()]
    
    for category_dir in sorted(categories):
        category_name = category_dir.name
        LOGGER.info("Generating manifest for MUSAN/%s", category_name)
        
        audio_files = list_audio_files(category_dir)
        
        if not audio_files:
            LOGGER.warning("No audio files in: %s", category_dir)
            continue
        
        rows: List[Dict[str, str]] = []
        
        for idx, audio_path in enumerate(
            tqdm(audio_files, desc=f"musan:{category_name}"), 
            start=1
        ):
            try:
                duration, sample_rate = get_audio_metadata_soundfile(audio_path)
            except Exception as exc:
                LOGGER.warning("Failed to process %s: %s", audio_path, exc)
                continue
            
            utt_id = f"{category_name}_{idx:08d}"
            
            rows.append({
                "dataset": "musan",
                "split": category_name,
                "utt_id": utt_id,
                "path": str(audio_path.absolute()),
                "duration_seconds": f"{duration:.6f}",
                "sampling_rate": str(sample_rate),
                "text": "",
                "speaker": "",
                "accent": "",
            })
        
        # Write manifest for this category
        manifest_path = manifest_dir / f"musan__{category_name}.csv"
        write_manifest(rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
        manifest_paths.append(manifest_path)
    
    return manifest_paths
