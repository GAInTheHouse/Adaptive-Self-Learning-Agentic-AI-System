#!/usr/bin/env python3
"""
Build per-subset LibriSpeech manifests from extracted OpenSLR SLR12 data.

Expected input layout example:
data/openslr/SLR12_LibriSpeech/LibriSpeech/dev-clean/...
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
from pathlib import Path
from typing import Dict, List


LOGGER = logging.getLogger("make_librispeech_manifest")
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


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _require_package(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing Python package '{module_name}'. Install with: pip install {module_name}"
        ) from exc


def _load_transcript_map(split_dir: Path) -> Dict[str, str]:
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


def _audio_metadata(audio_path: Path) -> tuple[str, str]:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return f"{float(info.duration):.6f}", str(int(info.samplerate))


def _build_rows_for_split(split_dir: Path) -> List[Dict[str, str]]:
    transcript_map = _load_transcript_map(split_dir)
    rows: List[Dict[str, str]] = []

    for flac_path in sorted(split_dir.rglob("*.flac")):
        utt_id = flac_path.stem
        duration, sampling_rate = _audio_metadata(flac_path)
        speaker = utt_id.split("-")[0] if "-" in utt_id else ""
        rows.append(
            {
                "dataset": "librispeech",
                "split": split_dir.name,
                "utt_id": utt_id,
                "path": str(flac_path),
                "duration_seconds": duration,
                "sampling_rate": sampling_rate,
                "text": transcript_map.get(utt_id, ""),
                "speaker": speaker,
                "accent": "",
            }
        )
    return rows


def _write_manifest(rows: List[Dict[str, str]], out_csv: Path, force: bool) -> None:
    if out_csv.exists() and not force:
        LOGGER.info("Manifest exists, skipping: %s", out_csv)
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    LOGGER.info("Wrote manifest: %s (%d rows)", out_csv, len(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-subset LibriSpeech manifests from extracted SLR12 directories."
    )
    parser.add_argument(
        "--librispeech-root",
        type=Path,
        default=Path("data") / "openslr" / "SLR12_LibriSpeech" / "LibriSpeech",
        help="Root containing LibriSpeech subset directories (dev-clean/dev-other/test-clean/test-other/etc.).",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data") / "manifests",
        help="Output directory for subset manifest CSV files.",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Optional subset names to process. If omitted, process all child directories in --librispeech-root.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest CSVs.")
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    _require_package("soundfile")
    args = parse_args()

    root = args.librispeech_root
    if not root.exists():
        raise RuntimeError(f"LibriSpeech root not found: {root}")

    if args.subsets:
        split_dirs = [root / subset for subset in args.subsets]
    else:
        split_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not split_dirs:
        raise RuntimeError(f"No LibriSpeech subset directories found under: {root}")

    for split_dir in split_dirs:
        if not split_dir.exists() or not split_dir.is_dir():
            raise RuntimeError(f"Subset directory does not exist: {split_dir}")
        rows = _build_rows_for_split(split_dir)
        out_csv = args.manifest_dir / f"librispeech__{split_dir.name}.csv"
        _write_manifest(rows, out_csv, force=args.force)

    LOGGER.info("Completed LibriSpeech manifest generation.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1)
