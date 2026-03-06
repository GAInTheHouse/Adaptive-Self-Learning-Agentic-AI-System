#!/usr/bin/env python3
"""
TED-LIUM Release 3 manifest generator.

Parses the TED-LIUM 3 "legacy" split layout:

    TEDLIUM_release-3/
      legacy/
        train/
          sph/   *.sph   (NIST sphere audio, 16kHz mono)
          stm/   *.stm   (segmentation + transcript)
        dev/   ...
        test/  ...

Each STM line represents one utterance segment within a full talk file.
The generated manifest rows reference the full SPH file path; segment
timing is encoded in the utt_id as ``<talk_id>-<segment_idx>`` and the
duration is derived from ``end_time - start_time`` in the STM file.

STM line format (space-separated):
    <talk_id> <channel> <speaker> <start_time> <end_time> <label> <words...>

Lines with text ``ignore_time_segment_in_scoring`` are skipped.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_utils import (
    CSV_FIELDS,
    write_manifest,
)

LOGGER = logging.getLogger("tedlium_generator")

TEDLIUM_SPLITS = ("train", "dev", "test")
TEDLIUM_SR = 16000


def _parse_stm_file(stm_path: Path) -> List[Tuple[str, str, float, float, str]]:
    """
    Parse a TED-LIUM STM file into segment tuples.

    Returns:
        List of (talk_id, speaker, start, end, text) tuples.
        Segments with ``ignore_time_segment_in_scoring`` text are excluded.
    """
    segments: List[Tuple[str, str, float, float, str]] = []

    with stm_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue

            parts = line.split(maxsplit=6)
            if len(parts) < 6:
                continue

            talk_id = parts[0]
            speaker = parts[2]
            try:
                start = float(parts[3])
                end = float(parts[4])
            except ValueError:
                LOGGER.warning("Could not parse times in STM line: %s", line[:80])
                continue

            text = parts[6].strip() if len(parts) > 6 else ""
            if text == "ignore_time_segment_in_scoring":
                continue

            segments.append((talk_id, speaker, start, end, text))

    return segments


def _sph_duration(sph_path: Path) -> Optional[float]:
    """Return duration of an SPH file via soundfile (falls back to None)."""
    try:
        import soundfile as sf
        info = sf.info(str(sph_path))
        return float(info.duration)
    except Exception as exc:
        LOGGER.debug("soundfile could not read %s: %s", sph_path.name, exc)
        return None


def build_rows_for_split(split_dir: Path, split_name: str) -> List[Dict[str, str]]:
    """
    Build manifest rows for one TED-LIUM split.

    Args:
        split_dir: Path to the split directory (contains sph/ and stm/).
        split_name: Name of the split (train / dev / test).

    Returns:
        List of manifest row dicts.
    """
    sph_dir = split_dir / "sph"
    stm_dir = split_dir / "stm"

    if not sph_dir.exists():
        LOGGER.warning("SPH directory not found: %s", sph_dir)
        return []
    if not stm_dir.exists():
        LOGGER.warning("STM directory not found: %s", stm_dir)
        return []

    # Build talk_id → sph_path mapping
    sph_map: Dict[str, Path] = {p.stem: p for p in sph_dir.glob("*.sph")}

    rows: List[Dict[str, str]] = []

    for stm_path in sorted(stm_dir.glob("*.stm")):
        segments = _parse_stm_file(stm_path)
        if not segments:
            continue

        # talk_id is consistent across segments in one STM file
        talk_id = stm_path.stem
        sph_path = sph_map.get(talk_id)

        if sph_path is None:
            LOGGER.warning("No SPH file for talk '%s'; skipping", talk_id)
            continue

        for idx, (seg_talk_id, speaker, start, end, text) in enumerate(segments):
            duration = round(end - start, 6)
            if duration <= 0:
                continue

            utt_id = f"{seg_talk_id}-{idx:05d}"

            rows.append({
                "dataset": "tedlium",
                "split": split_name,
                "utt_id": utt_id,
                "path": str(sph_path.absolute()),
                "duration_seconds": f"{duration:.6f}",
                "sampling_rate": str(TEDLIUM_SR),
                "text": text,
                "speaker": speaker,
                "accent": "",
            })

    LOGGER.info(
        "TED-LIUM %s: %d segments from %d talks",
        split_name, len(rows), len(sph_map),
    )
    return rows


def generate(data_dir: Path, manifest_dir: Path, force: bool) -> List[Path]:
    """
    Generate TED-LIUM 3 manifest CSV files.

    Expects the standard extraction layout::

        data_dir/legacy/train/sph/*.sph
        data_dir/legacy/train/stm/*.stm
        data_dir/legacy/dev/...
        data_dir/legacy/test/...

    If ``data_dir/legacy`` does not exist, ``data_dir`` itself is searched
    for split subdirectories directly.

    Args:
        data_dir: Root directory for the TEDLIUM_release-3 extraction.
        manifest_dir: Output directory for manifest CSV files.
        force: If True, overwrite existing manifests.

    Returns:
        List of generated manifest file paths.
    """
    manifest_paths: List[Path] = []

    legacy_dir = data_dir / "legacy"
    search_root = legacy_dir if legacy_dir.exists() else data_dir

    if not search_root.exists():
        LOGGER.warning("TED-LIUM root not found: %s", search_root)
        return manifest_paths

    found_any = False
    for split_name in TEDLIUM_SPLITS:
        split_dir = search_root / split_name
        if not split_dir.exists():
            LOGGER.debug("Split directory not found, skipping: %s", split_dir)
            continue

        LOGGER.info("Generating TED-LIUM manifest for split: %s", split_name)
        rows = build_rows_for_split(split_dir, split_name)

        if not rows:
            LOGGER.warning("No segments produced for TED-LIUM split: %s", split_name)
            continue

        found_any = True
        manifest_path = manifest_dir / f"tedlium__{split_name}.csv"
        write_manifest(rows, manifest_path, fieldnames=CSV_FIELDS, force=force)
        manifest_paths.append(manifest_path)

    if not found_any:
        LOGGER.warning(
            "No TED-LIUM split directories (train/dev/test) found under: %s", search_root
        )

    return manifest_paths
