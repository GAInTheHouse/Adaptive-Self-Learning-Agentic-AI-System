#!/usr/bin/env python3
"""
Bootstrap speech datasets, manifests, and derived audio variants.

Example:
    python scripts/bootstrap_data.py --all
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence

if TYPE_CHECKING:
    import numpy as np


LOGGER = logging.getLogger("bootstrap_data")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
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

OPENSLR_URLS = {
    "librispeech_dev_clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "librispeech_dev_other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "librispeech_test_clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "librispeech_test_other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "musan": "https://www.openslr.org/resources/17/musan.tar.gz",
    "rirs_noises": "https://www.openslr.org/resources/28/rirs_noises.zip",
    "tedlium3": "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
    "st_aeds": "https://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz",
}

HF_DATASETS = {
    "common_voice_en": {
        "name": "fsicoli/common_voice_17_0",
        "config": "en",
    },
    "voxpopuli_en": {
        "name": "facebook/voxpopuli",
        "config": "en",
    },
}

REQUIRED_COMMON_PYTHON_PACKAGES = ("tqdm",)
REQUIRED_HF_PACKAGES = ("datasets",)
REQUIRED_CORRUPTION_PACKAGES = ("numpy",)


@dataclass
class ManifestRow:
    dataset: str
    split: str
    utt_id: str
    path: str
    duration_seconds: str
    sampling_rate: str
    text: str = ""
    speaker: str = ""
    accent: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "utt_id": self.utt_id,
            "path": self.path,
            "duration_seconds": self.duration_seconds,
            "sampling_rate": self.sampling_rate,
            "text": self.text,
            "speaker": self.speaker,
            "accent": self.accent,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download speech datasets, generate manifests, and derive audio variants."
    )
    parser.add_argument("--all", action="store_true", help="Run every step.")
    parser.add_argument("--download-openslr", action="store_true", help="Download OpenSLR datasets.")
    parser.add_argument("--download-hf", action="store_true", help="Download Hugging Face datasets.")
    parser.add_argument("--download-primock", action="store_true", help="Download PriMock57 medical consultations.")
    parser.add_argument("--download-afrimedqa", action="store_true", help="Download AfriMed-QA medical QA dataset.")
    parser.add_argument("--generate-manifests", action="store_true", help="Generate source manifests.")
    parser.add_argument("--derive-low-audio", action="store_true", help="Create 8kHz mono variants.")
    parser.add_argument("--derive-corrupted", action="store_true", help="Create corrupted audio variants.")
    parser.add_argument("--check", action="store_true", help="Check required system dependencies.")
    parser.add_argument(
        "--include-librispeech-test",
        action="store_true",
        help="Also download and process LibriSpeech test-clean and test-other.",
    )
    parser.add_argument(
        "--include-tedlium",
        action="store_true",
        help="Also download TED-LIUM Release 3 (430h conversational talks).",
    )
    parser.add_argument(
        "--include-st-aeds",
        action="store_true",
        help="Also download ST-AEDS spontaneous speech dataset (4.7h).",
    )
    parser.add_argument(
        "--hf-max-samples",
        type=int,
        default=None,
        help="Optional max samples per HF split (for quick smoke runs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used for corruption operations.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    paths = {
        "base": base_dir,
        "openslr_root": base_dir / "openslr",
        "openslr_librispeech": base_dir / "openslr" / "SLR12_LibriSpeech",
        "openslr_musan": base_dir / "openslr" / "SLR17_MUSAN",
        "openslr_rirs": base_dir / "openslr" / "SLR28_RIRS_NOISES",
        "openslr_tedlium": base_dir / "openslr" / "SLR51_TEDLIUM",
        "openslr_st_aeds": base_dir / "openslr" / "SLR45_STAEDS",
        "hf_root": base_dir / "hf",
        "hf_common_voice": base_dir / "hf" / "common_voice_en",
        "hf_voxpopuli": base_dir / "hf" / "voxpopuli_en",
        "primock57": base_dir / "primock57",
        "afrimedqa": base_dir / "afrimedqa",
        "manifests": base_dir / "manifests",
        "derived_root": base_dir / "derived",
        "derived_low_audio": base_dir / "derived" / "low_audio",
        "derived_corrupted": base_dir / "derived" / "corrupted",
        "downloads_cache": base_dir / ".downloads",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fail_fast_missing_python_packages(packages: Sequence[str]) -> None:
    missing = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)

    if missing:
        pkg_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing Python package(s): {pkg_str}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def fail_fast_missing_binary(binary_name: str, install_hint: str) -> None:
    if shutil.which(binary_name) is None:
        raise RuntimeError(
            f"Missing required binary '{binary_name}'. {install_hint}"
        )


def _ffmpeg_install_hint() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "Recommended fix: brew install ffmpeg"
    if system == "linux":
        return (
            "Install ffmpeg using your package manager "
            "(examples: apt, yum/dnf, or pacman)."
        )
    return "Install ffmpeg and ensure it is available on PATH."


def check_environment() -> int:
    try:
        process = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        LOGGER.error("ffmpeg is not installed or not on PATH. %s", _ffmpeg_install_hint())
        return 1

    if process.returncode != 0:
        LOGGER.error("ffmpeg check failed. %s", process.stderr.strip() or _ffmpeg_install_hint())
        return 1

    first_line = process.stdout.splitlines()[0] if process.stdout else "ffmpeg detected"
    LOGGER.info("Dependency check passed: %s", first_line)
    return 0


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        LOGGER.info("Skip download (exists): %s", destination)
        return destination

    if force and destination.exists():
        destination.unlink()

    LOGGER.info("Downloading: %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "bootstrap-data/1.0"})
    try:
        with urllib.request.urlopen(request) as response:
            total = response.length or 0
            with destination.open("wb") as output, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    progress.update(len(chunk))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error downloading {url}: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error downloading {url}: {exc.reason}") from exc

    return destination


def extract_archive(archive_path: Path, destination_dir: Path, force: bool = False) -> None:
    archive_name = archive_path.name.lower()
    LOGGER.info("Extracting: %s", archive_path)

    if archive_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, mode="r:gz") as tar_ref:
            tar_ref.extractall(destination_dir)
        return
    if archive_name.endswith(".tar"):
        with tarfile.open(archive_path, mode="r:") as tar_ref:
            tar_ref.extractall(destination_dir)
        return
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, mode="r") as zip_ref:
            zip_ref.extractall(destination_dir)
        return

    raise RuntimeError(
        f"Unsupported archive format for {archive_path}. "
        "Expected .tar, .tar.gz, .tgz, or .zip."
    )


def download_openslr(
    paths: Dict[str, Path],
    include_librispeech_test: bool,
    include_tedlium: bool,
    include_st_aeds: bool,
    force: bool,
) -> None:
    downloads = [
        ("librispeech_dev_clean", paths["openslr_librispeech"], paths["openslr_librispeech"] / "LibriSpeech" / "dev-clean"),
        ("librispeech_dev_other", paths["openslr_librispeech"], paths["openslr_librispeech"] / "LibriSpeech" / "dev-other"),
        ("musan", paths["openslr_musan"], paths["openslr_musan"] / "musan"),
        ("rirs_noises", paths["openslr_rirs"], paths["openslr_rirs"] / "RIRS_NOISES"),
    ]
    if include_librispeech_test:
        downloads.extend(
            [
                ("librispeech_test_clean", paths["openslr_librispeech"], paths["openslr_librispeech"] / "LibriSpeech" / "test-clean"),
                ("librispeech_test_other", paths["openslr_librispeech"], paths["openslr_librispeech"] / "LibriSpeech" / "test-other"),
            ]
        )
    if include_tedlium:
        downloads.append(("tedlium3", paths["openslr_tedlium"], paths["openslr_tedlium"] / "TEDLIUM_release-3"))
    if include_st_aeds:
        downloads.append(("st_aeds", paths["openslr_st_aeds"], paths["openslr_st_aeds"] / "ST-AEDS-20180100_1-OS"))

    for key, extract_dir, expected_path in downloads:
        url = OPENSLR_URLS[key]
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = paths["downloads_cache"] / archive_name

        if expected_path.exists() and not force:
            LOGGER.info("Skip OpenSLR dataset (exists): %s", expected_path)
            continue

        if force and expected_path.exists():
            if expected_path.is_dir():
                shutil.rmtree(expected_path)
            else:
                expected_path.unlink()

        download_file(url, archive_path, force=force)
        extract_archive(archive_path, extract_dir, force=force)


def _safe_utt_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", value).strip("_")
    return cleaned or "utt"


def _audio_metadata(audio_path: Path) -> tuple[float, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=sample_rate",
        "-select_streams",
        "a:0",
        "-of",
        "json",
        str(audio_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {audio_path}.\n"
            f"stderr: {process.stderr.strip()}"
        )
    try:
        payload = json.loads(process.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload.get("streams", [])
        sample_rate = int(streams[0]["sample_rate"]) if streams else 0
        if sample_rate <= 0:
            raise ValueError("invalid sample_rate")
        return duration, sample_rate
    except Exception as exc:
        raise RuntimeError(f"Could not parse ffprobe output for {audio_path}.") from exc


def _list_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def _write_manifest(rows: Iterable[ManifestRow], out_path: Path, force: bool = False) -> None:
    if out_path.exists() and not force:
        LOGGER.info("Skip manifest (exists): %s", out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    LOGGER.info("Wrote manifest: %s", out_path)


def generate_librispeech_manifests(paths: Dict[str, Path], force: bool) -> None:
    from tqdm import tqdm

    libri_root = paths["openslr_librispeech"] / "LibriSpeech"
    if not libri_root.exists():
        LOGGER.warning("LibriSpeech not found, skipping manifests at: %s", libri_root)
        return

    splits = [p for p in libri_root.iterdir() if p.is_dir() and p.name.startswith(("dev-", "test-"))]
    for split_dir in sorted(splits):
        transcript_map: Dict[str, str] = {}
        for transcript_file in split_dir.rglob("*.trans.txt"):
            with transcript_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 1:
                        transcript_map[parts[0]] = ""
                    else:
                        transcript_map[parts[0]] = parts[1]

        audio_files = sorted(split_dir.rglob("*.flac"))
        rows: List[ManifestRow] = []
        for audio_path in tqdm(audio_files, desc=f"manifest:librispeech:{split_dir.name}"):
            utt_id = audio_path.stem
            duration, sample_rate = _audio_metadata(audio_path)
            rows.append(
                ManifestRow(
                    dataset="librispeech",
                    split=split_dir.name,
                    utt_id=utt_id,
                    path=str(audio_path),
                    duration_seconds=f"{duration:.6f}",
                    sampling_rate=str(sample_rate),
                    text=transcript_map.get(utt_id, ""),
                    speaker=utt_id.split("-")[0] if "-" in utt_id else "",
                )
            )

        out_csv = paths["manifests"] / f"librispeech__{split_dir.name}.csv"
        _write_manifest(rows, out_csv, force=force)


def generate_musan_manifests(paths: Dict[str, Path], force: bool) -> None:
    from tqdm import tqdm

    musan_root = paths["openslr_musan"] / "musan"
    if not musan_root.exists():
        LOGGER.warning("MUSAN not found, skipping manifests at: %s", musan_root)
        return

    categories = [p for p in musan_root.iterdir() if p.is_dir()]
    for category_dir in sorted(categories):
        audio_files = _list_audio_files(category_dir)
        rows: List[ManifestRow] = []
        for idx, audio_path in enumerate(tqdm(audio_files, desc=f"manifest:musan:{category_dir.name}"), start=1):
            duration, sample_rate = _audio_metadata(audio_path)
            utt_id = f"{category_dir.name}_{idx:08d}"
            rows.append(
                ManifestRow(
                    dataset="musan",
                    split=category_dir.name,
                    utt_id=utt_id,
                    path=str(audio_path),
                    duration_seconds=f"{duration:.6f}",
                    sampling_rate=str(sample_rate),
                )
            )

        out_csv = paths["manifests"] / f"musan__{category_dir.name}.csv"
        _write_manifest(rows, out_csv, force=force)


def generate_rirs_manifests(paths: Dict[str, Path], force: bool) -> None:
    from tqdm import tqdm

    rirs_root = paths["openslr_rirs"] / "RIRS_NOISES"
    if not rirs_root.exists():
        LOGGER.warning("RIRS_NOISES not found, skipping manifests at: %s", rirs_root)
        return

    audio_files = _list_audio_files(rirs_root)
    rows: List[ManifestRow] = []
    for idx, audio_path in enumerate(tqdm(audio_files, desc="manifest:rirs"), start=1):
        duration, sample_rate = _audio_metadata(audio_path)
        relative_parts = audio_path.relative_to(rirs_root).parts
        split = relative_parts[0] if relative_parts else "default"
        rows.append(
            ManifestRow(
                dataset="rirs_noises",
                split=split,
                utt_id=f"rirs_{idx:08d}",
                path=str(audio_path),
                duration_seconds=f"{duration:.6f}",
                sampling_rate=str(sample_rate),
            )
        )

    grouped: Dict[str, List[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.split, []).append(row)

    for split, split_rows in grouped.items():
        out_csv = paths["manifests"] / f"rirs_noises__{split}.csv"
        _write_manifest(split_rows, out_csv, force=force)


def _pick_field(example: Dict[str, object], candidates: Sequence[str]) -> str:
    for field in candidates:
        value = example.get(field)
        if value is None:
            continue
        return str(value)
    return ""


def _get_dataset_splits(dataset_name: str, config: str) -> List[str]:
    from datasets import get_dataset_split_names

    try:
        splits = get_dataset_split_names(dataset_name, config)
        if not splits:
            raise RuntimeError("No splits returned by get_dataset_split_names.")
        return list(splits)
    except Exception as exc:
        raise RuntimeError(
            f"Could not discover splits for Hugging Face dataset '{dataset_name}' ({config}). {exc}"
        ) from exc


def _save_hf_audio(
    audio_value: Dict[str, object],
    output_path: Path,
) -> tuple[float, int]:
    import numpy as np

    array = audio_value.get("array")
    sr = audio_value.get("sampling_rate")
    if array is None or sr is None:
        raise RuntimeError(
            "HF sample does not include decoded audio array/sampling_rate. "
            "This usually indicates a dataset schema mismatch."
        )

    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(int(sr)),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    process = subprocess.run(command, input=data.tobytes(), capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed writing HF audio to {output_path}.\n"
            f"stderr: {process.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return _audio_metadata(output_path)


def download_hf_dataset(
    hf_key: str,
    paths: Dict[str, Path],
    force: bool,
    max_samples_per_split: Optional[int],
) -> None:
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    config = HF_DATASETS[hf_key]
    dataset_name = config["name"]
    config_name = config["config"]
    output_root = paths["hf_common_voice"] if hf_key == "common_voice_en" else paths["hf_voxpopuli"]
    dataset_label = "common_voice_en" if hf_key == "common_voice_en" else "voxpopuli_en"

    split_names = _get_dataset_splits(dataset_name, config_name)
    LOGGER.info(
        "HF dataset %s (%s) splits: %s",
        dataset_name,
        config_name,
        ", ".join(split_names),
    )

    for split_name in split_names:
        split_out_dir = output_root / split_name
        manifest_path = paths["manifests"] / f"{dataset_label}__{split_name}.csv"

        if split_out_dir.exists() and manifest_path.exists() and not force:
            LOGGER.info("Skip HF split (exists): %s", split_name)
            continue

        if force and split_out_dir.exists():
            shutil.rmtree(split_out_dir)

        LOGGER.info("Downloading HF split: %s (%s)", dataset_label, split_name)
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split_name,
        )
        if "audio" not in dataset.column_names:
            raise RuntimeError(
                f"HF dataset '{dataset_name}' split '{split_name}' has no 'audio' column."
            )
        dataset = dataset.cast_column("audio", Audio(decode=True))

        total = len(dataset)
        if max_samples_per_split is not None:
            total = min(total, max_samples_per_split)
            dataset = dataset.select(range(total))
            LOGGER.info("Applying --hf-max-samples=%d for split %s", total, split_name)

        rows: List[ManifestRow] = []
        for idx, example in enumerate(tqdm(dataset, total=total, desc=f"hf:{dataset_label}:{split_name}")):
            utt_source = _pick_field(example, ["id", "utterance_id", "path", "client_id"])
            utt_id = _safe_utt_id(utt_source or f"{split_name}_{idx}")
            out_audio_path = split_out_dir / f"{utt_id}.wav"

            if out_audio_path.exists() and not force:
                duration, sample_rate = _audio_metadata(out_audio_path)
            else:
                duration, sample_rate = _save_hf_audio(example["audio"], out_audio_path)

            rows.append(
                ManifestRow(
                    dataset=dataset_label,
                    split=split_name,
                    utt_id=utt_id,
                    path=str(out_audio_path),
                    duration_seconds=f"{duration:.6f}",
                    sampling_rate=str(sample_rate),
                    text=_pick_field(example, ["sentence", "normalized_text", "transcription"]),
                    speaker=_pick_field(example, ["client_id", "speaker_id", "speaker"]),
                    accent=_pick_field(example, ["accent", "variant"]),
                )
            )

        _write_manifest(rows, manifest_path, force=force)


def generate_source_manifests(paths: Dict[str, Path], force: bool) -> None:
    generate_librispeech_manifests(paths, force=force)
    generate_musan_manifests(paths, force=force)
    generate_rirs_manifests(paths, force=force)


def _load_source_manifest_rows(manifests_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in sorted(manifests_dir.glob("*.csv")):
        name = csv_path.stem
        if name.startswith("low_audio__") or name.startswith("corrupted__"):
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("path", "")
                if not path:
                    continue
                audio_path = Path(path)
                if audio_path.exists():
                    rows.append(row)
    return rows


def _relative_audio_output_path(row: Dict[str, str], extension: str = ".wav") -> Path:
    dataset = _safe_utt_id(row.get("dataset", "dataset"))
    split = _safe_utt_id(row.get("split", "split"))
    utt_id = _safe_utt_id(row.get("utt_id", "utt"))
    return Path(dataset) / split / f"{utt_id}{extension}"


def derive_low_audio(paths: Dict[str, Path], force: bool) -> None:
    from tqdm import tqdm

    fail_fast_missing_binary(
        "ffmpeg",
        _ffmpeg_install_hint(),
    )

    source_rows = _load_source_manifest_rows(paths["manifests"])
    if not source_rows:
        LOGGER.warning("No source manifests found for low-audio derivation.")
        return

    grouped_rows: Dict[tuple[str, str], List[ManifestRow]] = {}
    for row in tqdm(source_rows, desc="derive:low_audio"):
        src_path = Path(row["path"])
        rel_path = _relative_audio_output_path(row)
        out_path = paths["derived_low_audio"] / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not out_path.exists() or force:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src_path),
                "-ac",
                "1",
                "-ar",
                "8000",
                "-sample_fmt",
                "s16",
                str(out_path),
            ]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed for {src_path} -> {out_path}.\n"
                    f"stderr: {process.stderr.strip()}"
                )

        duration, sample_rate = _audio_metadata(out_path)
        manifest_row = ManifestRow(
            dataset=row.get("dataset", ""),
            split=row.get("split", ""),
            utt_id=row.get("utt_id", ""),
            path=str(out_path),
            duration_seconds=f"{duration:.6f}",
            sampling_rate=str(sample_rate),
            text=row.get("text", ""),
            speaker=row.get("speaker", ""),
            accent=row.get("accent", ""),
        )
        key = (manifest_row.dataset, manifest_row.split)
        grouped_rows.setdefault(key, []).append(manifest_row)

    for (dataset, split), rows in grouped_rows.items():
        manifest_name = f"low_audio__{_safe_utt_id(dataset)}__{_safe_utt_id(split)}.csv"
        _write_manifest(rows, paths["manifests"] / manifest_name, force=True)


def _load_audio_mono_float32(path: Path, target_sr: Optional[int] = None) -> tuple["np.ndarray", int]:
    import numpy as np

    _, native_sr = _audio_metadata(path)
    output_sr = int(target_sr) if target_sr is not None else native_sr
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(output_sr),
        "pipe:1",
    ]
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed decoding {path}.\n"
            f"stderr: {process.stderr.decode('utf-8', errors='replace').strip()}"
        )
    audio = np.frombuffer(process.stdout, dtype=np.float32).copy()
    return audio, output_sr


def _choose_noise_segment(noise: "np.ndarray", target_len: int, rng: random.Random) -> "np.ndarray":
    import numpy as np

    if len(noise) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(noise) >= target_len:
        start = rng.randint(0, len(noise) - target_len)
        return noise[start : start + target_len]

    repeats = int(np.ceil(target_len / len(noise)))
    tiled = np.tile(noise, repeats)
    return tiled[:target_len]


def _mix_additive_noise(
    clean: "np.ndarray",
    noise: "np.ndarray",
    snr_db: float,
) -> "np.ndarray":
    import numpy as np

    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return clean + noise * float(scale)


def _apply_reverb(clean: "np.ndarray", rir: "np.ndarray") -> "np.ndarray":
    import numpy as np

    if len(rir) == 0:
        return clean
    rir = rir / (max(abs(rir).max(), 1e-8))
    reverbed = np.convolve(clean, rir, mode="full")
    return reverbed[: len(clean)].astype(np.float32)


def _apply_random_dropouts(audio: "np.ndarray", sr: int, rng: random.Random) -> "np.ndarray":
    out = audio.copy()
    n_dropouts = rng.randint(1, 4)
    for _ in range(n_dropouts):
        duration_s = rng.uniform(0.01, 0.08)
        drop_len = max(1, int(duration_s * sr))
        if drop_len >= len(out):
            break
        start = rng.randint(0, len(out) - drop_len)
        out[start : start + drop_len] = 0.0
    return out


def _apply_random_clipping(audio: "np.ndarray", rng: random.Random) -> "np.ndarray":
    import numpy as np

    gain = rng.uniform(0.9, 1.5)
    clipped = np.clip(audio * gain, -1.0, 1.0)
    return clipped.astype(np.float32)


def _write_pcm16_wav(path: Path, audio: "np.ndarray", sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    process = subprocess.run(command, input=audio.astype("float32").tobytes(), capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed writing audio to {path}.\n"
            f"stderr: {process.stderr.decode('utf-8', errors='replace').strip()}"
        )


def _load_musan_noise_files(paths: Dict[str, Path]) -> List[Path]:
    musan_root = paths["openslr_musan"] / "musan" / "noise"
    if not musan_root.exists():
        return []
    return _list_audio_files(musan_root)


def _load_rir_files(paths: Dict[str, Path]) -> List[Path]:
    rirs_root = paths["openslr_rirs"] / "RIRS_NOISES"
    if not rirs_root.exists():
        return []
    return _list_audio_files(rirs_root)


def derive_corrupted_audio(paths: Dict[str, Path], force: bool, seed: int) -> None:
    from tqdm import tqdm

    fail_fast_missing_python_packages(REQUIRED_CORRUPTION_PACKAGES)
    source_rows = _load_source_manifest_rows(paths["manifests"])
    if not source_rows:
        LOGGER.warning("No source manifests found for corruption derivation.")
        return

    noise_files = _load_musan_noise_files(paths)
    if not noise_files:
        raise RuntimeError(
            "No MUSAN noise files found. Expected audio under "
            f"{paths['openslr_musan'] / 'musan' / 'noise'}."
        )

    rir_files = _load_rir_files(paths)
    if not rir_files:
        LOGGER.warning("No RIRS files found; corruption will run without reverb.")

    rng = random.Random(seed)
    grouped_rows: Dict[tuple[str, str], List[ManifestRow]] = {}

    for row in tqdm(source_rows, desc="derive:corrupted"):
        src_path = Path(row["path"])
        rel_path = _relative_audio_output_path(row)
        out_path = paths["derived_corrupted"] / rel_path

        if out_path.exists() and not force:
            duration, sample_rate = _audio_metadata(out_path)
        else:
            clean, clean_sr = _load_audio_mono_float32(src_path)
            noise_path = rng.choice(noise_files)
            noise, _ = _load_audio_mono_float32(noise_path, target_sr=clean_sr)
            noise_seg = _choose_noise_segment(noise, len(clean), rng)
            snr_db = rng.uniform(5.0, 20.0)
            corrupted = _mix_additive_noise(clean, noise_seg, snr_db)

            # Optional RIRS convolution on ~50% samples.
            if rir_files and rng.random() < 0.5:
                rir_path = rng.choice(rir_files)
                rir, _ = _load_audio_mono_float32(rir_path, target_sr=clean_sr)
                corrupted = _apply_reverb(corrupted, rir)

            corrupted = _apply_random_clipping(corrupted, rng)
            corrupted = _apply_random_dropouts(corrupted, clean_sr, rng)
            _write_pcm16_wav(out_path, corrupted, clean_sr)
            duration, sample_rate = _audio_metadata(out_path)

        manifest_row = ManifestRow(
            dataset=row.get("dataset", ""),
            split=row.get("split", ""),
            utt_id=row.get("utt_id", ""),
            path=str(out_path),
            duration_seconds=f"{duration:.6f}",
            sampling_rate=str(sample_rate),
            text=row.get("text", ""),
            speaker=row.get("speaker", ""),
            accent=row.get("accent", ""),
        )
        grouped_rows.setdefault((manifest_row.dataset, manifest_row.split), []).append(manifest_row)

    for (dataset, split), rows in grouped_rows.items():
        manifest_name = f"corrupted__{_safe_utt_id(dataset)}__{_safe_utt_id(split)}.csv"
        _write_manifest(rows, paths["manifests"] / manifest_name, force=True)


def download_primock57(paths: Dict[str, Path], force: bool) -> None:
    """Download PriMock57 dataset using external script."""
    LOGGER.info("Downloading PriMock57 medical consultation dataset...")
    
    script_path = Path(__file__).parent / "primock_download.py"
    if not script_path.exists():
        LOGGER.warning("primock_download.py not found, skipping PriMock57 download")
        return
    
    cmd = [
        sys.executable,
        str(script_path),
        "--out-dir", str(paths["primock57"]),
        "--manifest-dir", str(paths["manifests"]),
    ]
    
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.warning("PriMock57 download failed (check git-lfs installation)")


def download_afrimedqa(paths: Dict[str, Path], force: bool) -> None:
    """Download AfriMed-QA dataset using external script."""
    LOGGER.info("Downloading AfriMed-QA medical QA dataset...")
    
    script_path = Path(__file__).parent / "afrimedqa_download.py"
    if not script_path.exists():
        LOGGER.warning("afrimedqa_download.py not found, skipping AfriMed-QA download")
        return
    
    cmd = [
        sys.executable,
        str(script_path),
        "--out-dir", str(paths["afrimedqa"]),
        "--manifest-dir", str(paths["manifests"]),
    ]
    
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.warning("AfriMed-QA download failed")


def _resolve_plan(args: argparse.Namespace) -> Dict[str, bool]:
    return {
        "download_openslr": args.all or args.download_openslr,
        "download_hf": args.all or args.download_hf,
        "download_primock": args.all or args.download_primock,
        "download_afrimedqa": args.all or args.download_afrimedqa,
        "generate_manifests": args.all or args.generate_manifests,
        "derive_low_audio": args.all or args.derive_low_audio,
        "derive_corrupted": args.all or args.derive_corrupted,
    }


def main() -> int:
    args = parse_args()
    configure_logging()

    if args.check:
        return check_environment()

    if not any(
        (
            args.all,
            args.download_openslr,
            args.download_hf,
            args.download_primock,
            args.download_afrimedqa,
            args.generate_manifests,
            args.derive_low_audio,
            args.derive_corrupted,
        )
    ):
        LOGGER.error("No operation selected. Use --all or choose specific flags.")
        return 2

    fail_fast_missing_python_packages(REQUIRED_COMMON_PYTHON_PACKAGES)
    fail_fast_missing_binary("ffmpeg", _ffmpeg_install_hint())
    fail_fast_missing_binary("ffprobe", "Install ffmpeg package; ffprobe is bundled with it.")

    base_dir = Path("data")
    paths = ensure_directories(base_dir)
    plan = _resolve_plan(args)

    LOGGER.info("Bootstrap plan: %s", ", ".join([k for k, v in plan.items() if v]))
    LOGGER.info("Base data dir: %s", paths["base"].resolve())

    if plan["download_openslr"]:
        download_openslr(
            paths=paths,
            include_librispeech_test=args.include_librispeech_test,
            include_tedlium=args.include_tedlium,
            include_st_aeds=args.include_st_aeds,
            force=args.force,
        )

    if plan["download_hf"]:
        fail_fast_missing_python_packages(REQUIRED_HF_PACKAGES)
        download_hf_dataset(
            hf_key="common_voice_en",
            paths=paths,
            force=args.force,
            max_samples_per_split=args.hf_max_samples,
        )
        download_hf_dataset(
            hf_key="voxpopuli_en",
            paths=paths,
            force=args.force,
            max_samples_per_split=args.hf_max_samples,
        )

    if plan["download_primock"]:
        download_primock57(paths=paths, force=args.force)

    if plan["download_afrimedqa"]:
        download_afrimedqa(paths=paths, force=args.force)

    if plan["generate_manifests"]:
        generate_source_manifests(paths=paths, force=args.force)

    if plan["derive_low_audio"]:
        derive_low_audio(paths=paths, force=args.force)

    if plan["derive_corrupted"]:
        derive_corrupted_audio(paths=paths, force=args.force, seed=args.seed)

    LOGGER.info("Bootstrap completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1)
