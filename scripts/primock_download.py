#!/usr/bin/env python3
"""
PriMock57 dataset downloader with Git LFS support.

Downloads the PriMock57 medical consultation dataset from GitHub, which contains:
- 57 mock medical consultations with audio recordings
- Manual utterance-level transcriptions
- Consultation notes written by clinicians

Source: https://github.com/babylonhealth/primock57
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOGGER = logging.getLogger("primock_download")

PRIMOCK_REPO_URL = "https://github.com/babylonhealth/primock57.git"

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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def check_git_lfs() -> bool:
    """Check if git-lfs is installed and initialized."""
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            LOGGER.info("Git LFS detected: %s", result.stdout.strip().split('\n')[0])
            return True
    except FileNotFoundError:
        pass
    
    LOGGER.error(
        "Git LFS is not installed. Install it to download PriMock57 audio files.\n"
        "macOS: brew install git-lfs\n"
        "Ubuntu/Debian: sudo apt install git-lfs\n"
        "After installation, run: git lfs install"
    )
    return False


def check_ffprobe() -> bool:
    """Check if ffprobe is available for audio metadata extraction."""
    if shutil.which("ffprobe") is None:
        LOGGER.error(
            "ffprobe is not installed. Install ffmpeg package.\n"
            "macOS: brew install ffmpeg\n"
            "Ubuntu/Debian: sudo apt install ffmpeg"
        )
        return False
    return True


def clone_repository(repo_url: str, target_dir: Path, force: bool = False) -> bool:
    """Clone the PriMock57 repository using git with LFS support."""
    if target_dir.exists():
        if not force:
            LOGGER.info("Repository already exists at: %s", target_dir)
            return True
        LOGGER.info("Removing existing repository (--force enabled)")
        shutil.rmtree(target_dir)
    
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("Cloning PriMock57 repository (this may take a while for LFS files)...")
    try:
        result = subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            LOGGER.error("Git clone failed:\n%s", result.stderr)
            return False
        
        LOGGER.info("Repository cloned successfully to: %s", target_dir)
        
        # Verify LFS files were downloaded
        lfs_check = subprocess.run(
            ["git", "lfs", "ls-files"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        
        if lfs_check.returncode == 0 and lfs_check.stdout.strip():
            lfs_count = len(lfs_check.stdout.strip().split('\n'))
            LOGGER.info("Git LFS files downloaded: %d files", lfs_count)
        
        return True
        
    except Exception as exc:
        LOGGER.error("Failed to clone repository: %s", exc)
        return False


def get_audio_metadata(audio_path: Path) -> Tuple[float, int]:
    """Extract duration and sample rate from audio file using ffprobe."""
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
        LOGGER.warning("Failed to get metadata for %s: %s", audio_path.name, exc)
        return 0.0, 16000


def parse_transcript(transcript_path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse a transcript file and extract utterances.
    
    Returns list of (speaker, text, timestamp) tuples.
    """
    utterances = []
    
    if not transcript_path.exists():
        return utterances
    
    try:
        with transcript_path.open("r", encoding="utf-8") as f:
            content = f.read()
        
        # Try to parse structured transcript format
        # PriMock57 transcripts may have various formats
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract speaker and text
            # Format could be "Speaker: text" or similar
            match = re.match(r'^([^:]+):\s*(.+)$', line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                utterances.append((speaker, text, ""))
            else:
                # If no speaker marker, treat as continuation or unknown speaker
                utterances.append(("unknown", line, ""))
        
        # If no structured format found, return full text as single utterance
        if not utterances and content.strip():
            utterances.append(("unknown", content.strip(), ""))
            
    except Exception as exc:
        LOGGER.warning("Failed to parse transcript %s: %s", transcript_path.name, exc)
    
    return utterances


def read_consultation_note(note_path: Path) -> str:
    """Read consultation note if available."""
    if not note_path.exists():
        return ""
    
    try:
        with note_path.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as exc:
        LOGGER.warning("Failed to read note %s: %s", note_path.name, exc)
        return ""


def generate_manifests(primock_dir: Path, manifest_dir: Path, force: bool = False) -> None:
    """
    Generate manifest CSV files from PriMock57 data.
    
    Scans audio/, transcripts/, and notes/ directories to create unified manifest.
    """
    from tqdm import tqdm
    
    audio_dir = primock_dir / "audio"
    transcripts_dir = primock_dir / "transcripts"
    notes_dir = primock_dir / "notes"
    
    if not audio_dir.exists():
        raise RuntimeError(f"Audio directory not found: {audio_dir}")
    
    # Find all audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    
    if not audio_files:
        LOGGER.warning("No WAV files found in %s", audio_dir)
        return
    
    LOGGER.info("Found %d audio files", len(audio_files))
    
    manifest_rows = []
    
    for audio_path in tqdm(audio_files, desc="Processing PriMock57 consultations"):
        # Extract consultation ID from filename (e.g., "consultation_01.wav" -> "01")
        consultation_id = audio_path.stem
        
        # Get audio metadata
        try:
            duration, sample_rate = get_audio_metadata(audio_path)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", audio_path.name, exc)
            continue
        
        # Find corresponding transcript
        transcript_path = transcripts_dir / f"{consultation_id}.txt"
        if not transcript_path.exists():
            # Try alternative naming patterns
            for pattern in ["*.txt", "*.trans.txt"]:
                matches = list(transcripts_dir.glob(pattern))
                if matches:
                    # Try to match by ID
                    for match in matches:
                        if consultation_id in match.stem:
                            transcript_path = match
                            break
        
        # Parse transcript to get full text
        utterances = parse_transcript(transcript_path)
        full_text = " ".join([utt[1] for utt in utterances])
        
        # Extract speakers
        speakers = list(set([utt[0] for utt in utterances if utt[0] != "unknown"]))
        speaker_str = ",".join(speakers) if speakers else ""
        
        # Read consultation note
        note_path = notes_dir / f"{consultation_id}.txt"
        
        # Create manifest entry
        manifest_rows.append({
            "dataset": "primock57",
            "split": "full",  # PriMock57 doesn't have predefined splits
            "utt_id": consultation_id,
            "path": str(audio_path.absolute()),
            "duration_seconds": f"{duration:.6f}",
            "sampling_rate": str(sample_rate),
            "text": full_text,
            "speaker": speaker_str,
            "accent": "uk",  # PriMock57 is from UK-based Babylon Health
        })
    
    # Write manifest
    manifest_path = manifest_dir / "primock57__full.csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    if manifest_path.exists() and not force:
        LOGGER.info("Manifest already exists: %s", manifest_path)
        return
    
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    
    LOGGER.info("Wrote manifest with %d entries: %s", len(manifest_rows), manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PriMock57 medical consultation dataset from GitHub."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "primock57",
        help="Output directory for cloned repository.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data") / "manifests",
        help="Directory for manifest CSV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing data.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip manifest generation (only download repository).",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    
    # Check prerequisites
    if not check_git_lfs():
        return 1
    
    if not check_ffprobe():
        return 1
    
    # Clone repository
    if not clone_repository(PRIMOCK_REPO_URL, args.out_dir, force=args.force):
        return 1
    
    # Generate manifests
    if not args.skip_manifest:
        try:
            # Check for tqdm
            import tqdm
            generate_manifests(args.out_dir, args.manifest_dir, force=args.force)
        except ImportError:
            LOGGER.error("tqdm package required for manifest generation. Install with: pip install tqdm")
            return 1
        except Exception as exc:
            LOGGER.error("Failed to generate manifests: %s", exc)
            return 1
    
    LOGGER.info("PriMock57 download completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.info("Download interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("Unexpected error: %s", exc)
        raise SystemExit(1)
