#!/usr/bin/env python3
"""
Download Hugging Face datasets, persist locally, and export per-split manifests.

Audio behavior:
- If a sample already references a local audio file path, keep that path.
- If audio is decoded/streamed, materialize to WAV under:
  data/hf_audio/<dataset>/<split>/
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


LOGGER = logging.getLogger("hf_download")
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


def _require_package(module_name: str, pip_name: Optional[str] = None) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        install_name = pip_name or module_name
        raise RuntimeError(
            f"Missing Python package '{install_name}'. Install with: pip install {install_name}"
        ) from exc


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def _pick_field(example: Dict[str, object], candidates: List[str]) -> str:
    for key in candidates:
        value = example.get(key)
        if value is None:
            continue
        return str(value)
    return ""


def _audio_metadata(path: Path) -> tuple[str, str]:
    import soundfile as sf

    info = sf.info(str(path))
    return f"{float(info.duration):.6f}", str(int(info.samplerate))


def _materialize_decoded_audio_to_wav(
    audio_value: Dict[str, object],
    out_path: Path,
) -> tuple[str, str]:
    import numpy as np
    import soundfile as sf

    data = audio_value.get("array")
    sr = audio_value.get("sampling_rate")
    if data is None or sr is None:
        raise RuntimeError("Decoded audio is missing 'array' or 'sampling_rate'.")

    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, int(sr), subtype="PCM_16")
    duration = float(len(audio)) / float(sr) if sr else 0.0
    return f"{duration:.6f}", str(int(sr))


def _resolve_audio_path_for_example(
    example: Dict[str, object],
    audio_column: str,
    out_audio_dir: Path,
    utt_id: str,
    force: bool,
) -> tuple[str, str, str]:
    audio_value = example.get(audio_column)
    if audio_value is None:
        return "", "", ""

    # Some datasets expose local path strings directly.
    if isinstance(audio_value, str):
        local_path = Path(audio_value)
        if local_path.exists():
            duration, sr = _audio_metadata(local_path)
            return str(local_path), duration, sr
        return audio_value, "", ""

    if not isinstance(audio_value, dict):
        return "", "", ""

    # If local file path already exists, keep it.
    candidate_path = audio_value.get("path")
    if isinstance(candidate_path, str):
        local_path = Path(candidate_path)
        if local_path.exists():
            duration, sr = _audio_metadata(local_path)
            return str(local_path), duration, sr

    # Otherwise materialize decoded audio.
    out_path = out_audio_dir / f"{utt_id}.wav"
    if out_path.exists() and not force:
        duration, sr = _audio_metadata(out_path)
        return str(out_path), duration, sr

    duration, sr = _materialize_decoded_audio_to_wav(audio_value, out_path)
    return str(out_path), duration, sr


def _detect_audio_column(dataset) -> Optional[str]:
    from datasets import Audio

    for col, feature in dataset.features.items():
        if isinstance(feature, Audio):
            return col
    if "audio" in dataset.column_names:
        return "audio"
    return None


def _dataset_slug(dataset_name: str, config: Optional[str]) -> str:
    if config:
        return _safe_name(f"{dataset_name}__{config}")
    return _safe_name(dataset_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face dataset splits, save_to_disk, and write manifests."
    )
    parser.add_argument("--dataset", required=True, help="Dataset id, e.g. mozilla-foundation/common_voice_17_0")
    parser.add_argument("--config", default=None, help="Dataset config/name, e.g. en")
    parser.add_argument(
        "--split",
        default="all",
        help="Specific split name (e.g. train) or 'all' to process all available splits.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data") / "hf_cache",
        help="Hugging Face cache directory passed to load_dataset.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data") / "hf_saved",
        help="Root directory for dataset.save_to_disk outputs.",
    )
    parser.add_argument(
        "--audio-out-dir",
        type=Path,
        default=Path("data") / "hf_audio",
        help="Root directory for materialized WAV audio.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data") / "manifests",
        help="Directory for per-split manifest CSV files.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit per split.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing saved splits/audio/manifests.")
    return parser.parse_args()


def _get_splits(dataset_name: str, config: Optional[str]) -> List[str]:
    from datasets import get_dataset_split_names

    splits = get_dataset_split_names(dataset_name, config)
    if not splits:
        raise RuntimeError(f"No splits found for dataset={dataset_name}, config={config}.")
    return list(splits)


def _save_split_dataset(dataset, split_save_dir: Path, force: bool) -> None:
    if split_save_dir.exists():
        if not force:
            LOGGER.info("save_to_disk exists, skipping: %s", split_save_dir)
            return
        shutil.rmtree(split_save_dir)

    split_save_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(split_save_dir))
    LOGGER.info("Saved split with save_to_disk: %s", split_save_dir)


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
    LOGGER.info("Wrote manifest: %s", out_csv)


def process_split(
    dataset_name: str,
    config: Optional[str],
    split_name: str,
    cache_dir: Path,
    save_dir: Path,
    audio_out_dir: Path,
    manifest_dir: Path,
    max_samples: Optional[int],
    force: bool,
) -> None:
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    LOGGER.info("Loading split '%s' for dataset=%s config=%s", split_name, dataset_name, config)
    split_ds = load_dataset(
        dataset_name,
        name=config,
        split=split_name,
        cache_dir=str(cache_dir),
    )

    if max_samples is not None:
        n = min(max_samples, len(split_ds))
        split_ds = split_ds.select(range(n))
        LOGGER.info("Using max_samples=%d for split '%s'", n, split_name)

    # Hugging Face guidance for audio columns: cast to Audio when needed.
    audio_column = _detect_audio_column(split_ds)
    if audio_column:
        try:
            split_ds = split_ds.cast_column(audio_column, Audio())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to cast audio column '{audio_column}' to datasets.Audio for split '{split_name}'."
            ) from exc

    dataset_slug = _dataset_slug(dataset_name, config)
    split_slug = _safe_name(split_name)

    split_save_dir = save_dir / dataset_slug / split_slug
    _save_split_dataset(split_ds, split_save_dir, force=force)

    target_audio_dir = audio_out_dir / dataset_slug / split_slug
    rows: List[Dict[str, str]] = []
    for idx, example in enumerate(tqdm(split_ds, desc=f"{dataset_slug}:{split_slug}", total=len(split_ds))):
        utt_src = _pick_field(example, ["id", "utterance_id", "path", "client_id"]) or f"{split_slug}_{idx}"
        utt_id = _safe_name(utt_src)

        path_str = ""
        duration_str = ""
        sampling_rate_str = ""
        if audio_column:
            path_str, duration_str, sampling_rate_str = _resolve_audio_path_for_example(
                example=example,
                audio_column=audio_column,
                out_audio_dir=target_audio_dir,
                utt_id=utt_id,
                force=force,
            )
        elif "path" in example:
            p = Path(str(example["path"]))
            path_str = str(p)
            if p.exists():
                duration_str, sampling_rate_str = _audio_metadata(p)

        rows.append(
            {
                "dataset": dataset_slug,
                "split": split_name,
                "utt_id": utt_id,
                "path": path_str,
                "duration_seconds": duration_str,
                "sampling_rate": sampling_rate_str,
                "text": _pick_field(example, ["sentence", "text", "normalized_text", "transcription"]),
                "speaker": _pick_field(example, ["speaker_id", "speaker", "client_id"]),
                "accent": _pick_field(example, ["accent", "variant"]),
            }
        )

    manifest_path = manifest_dir / f"{dataset_slug}__{split_slug}.csv"
    _write_manifest(rows, manifest_path, force=force)


def main() -> int:
    _configure_logging()
    args = parse_args()

    _require_package("datasets")
    _require_package("tqdm")
    _require_package("soundfile")
    _require_package("numpy")

    splits = _get_splits(args.dataset, args.config) if args.split == "all" else [args.split]
    LOGGER.info("Target splits: %s", ", ".join(splits))

    for split_name in splits:
        process_split(
            dataset_name=args.dataset,
            config=args.config,
            split_name=split_name,
            cache_dir=args.cache_dir,
            save_dir=args.save_dir,
            audio_out_dir=args.audio_out_dir,
            manifest_dir=args.manifest_dir,
            max_samples=args.max_samples,
            force=args.force,
        )

    LOGGER.info("Hugging Face download workflow completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1)
