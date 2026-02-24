#!/usr/bin/env python3
"""
Audio augmentation utility using canonical open corpora:
- MUSAN for additive noise
- RIRS_NOISES for room impulse responses (optional reverb)

Augmentations:
- Additive noise at random SNR from {0, 5, 10, 15, 20} dB (configurable)
- Optional random RIR convolution
- Random dropouts (zeroed chunks)
- Random clipping
"""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple


LOGGER = logging.getLogger("augment_audio")
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
DEFAULT_SNRS = (0.0, 5.0, 10.0, 15.0, 20.0)


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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def _list_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def _load_audio(path: Path, target_sr: int | None = None) -> Tuple["np.ndarray", int]:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def _write_audio(path: Path, audio: "np.ndarray", sr: int) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")


def _pick_noise_segment(noise: "np.ndarray", target_len: int, rng: random.Random) -> "np.ndarray":
    import numpy as np

    if len(noise) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(noise) >= target_len:
        start = rng.randint(0, len(noise) - target_len)
        return noise[start : start + target_len]
    repeats = int(np.ceil(target_len / len(noise)))
    return np.tile(noise, repeats)[:target_len]


def _mix_at_snr(clean: "np.ndarray", noise: "np.ndarray", snr_db: float) -> "np.ndarray":
    import numpy as np

    clean_power = float(np.mean(clean**2)) + 1e-12
    noise_power = float(np.mean(noise**2)) + 1e-12
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = (target_noise_power / noise_power) ** 0.5
    return (clean + noise * float(scale)).astype(np.float32)


def _apply_reverb(clean: "np.ndarray", rir: "np.ndarray") -> "np.ndarray":
    import numpy as np

    if len(rir) == 0:
        return clean
    normalized_rir = rir / (max(float(np.max(np.abs(rir))), 1e-8))
    reverbed = np.convolve(clean, normalized_rir, mode="full")
    return reverbed[: len(clean)].astype(np.float32)


def _apply_dropouts(audio: "np.ndarray", sr: int, rng: random.Random) -> "np.ndarray":
    out = audio.copy()
    n_segments = rng.randint(1, 4)
    for _ in range(n_segments):
        duration_s = rng.uniform(0.01, 0.08)
        n = max(1, int(duration_s * sr))
        if n >= len(out):
            continue
        start = rng.randint(0, len(out) - n)
        out[start : start + n] = 0.0
    return out


def _apply_clipping(audio: "np.ndarray", rng: random.Random) -> "np.ndarray":
    import numpy as np

    gain = rng.uniform(0.9, 1.5)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def _augment_one(
    src_audio: Path,
    noise_files: Sequence[Path],
    rir_files: Sequence[Path],
    out_path: Path,
    rng: random.Random,
    snr_choices: Sequence[float],
    reverb_probability: float,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        return

    clean, sr = _load_audio(src_audio)
    noise_path = rng.choice(noise_files)
    noise, _ = _load_audio(noise_path, target_sr=sr)
    noise_seg = _pick_noise_segment(noise, len(clean), rng)

    snr_db = rng.choice(list(snr_choices))
    augmented = _mix_at_snr(clean, noise_seg, snr_db)

    if rir_files and rng.random() < reverb_probability:
        rir_path = rng.choice(rir_files)
        rir, _ = _load_audio(rir_path, target_sr=sr)
        augmented = _apply_reverb(augmented, rir)

    augmented = _apply_dropouts(augmented, sr, rng)
    augmented = _apply_clipping(augmented, rng)
    _write_audio(out_path, augmented, sr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment audio using MUSAN noise, optional RIRS reverb, dropouts, and clipping."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source audio to augment.",
    )
    parser.add_argument(
        "--musan-dir",
        type=Path,
        default=Path("data") / "openslr" / "SLR17_MUSAN" / "musan" / "noise",
        help="MUSAN noise directory (canonical open augmentation corpus).",
    )
    parser.add_argument(
        "--rirs-dir",
        type=Path,
        default=Path("data") / "openslr" / "SLR28_RIRS_NOISES" / "RIRS_NOISES",
        help="RIRS directory (canonical open augmentation corpus).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "derived" / "corrupted",
        help="Directory to write augmented audio.",
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=list(DEFAULT_SNRS),
        help="SNR choices in dB (default: 0 5 10 15 20).",
    )
    parser.add_argument(
        "--reverb-probability",
        type=float,
        default=0.5,
        help="Probability of applying random RIRS convolution (0.0-1.0).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    _require_package("numpy")
    _require_package("librosa")
    _require_package("soundfile")

    args = parse_args()
    if not args.input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {args.input_dir}")
    if not args.musan_dir.exists():
        raise RuntimeError(f"MUSAN directory does not exist: {args.musan_dir}")
    if not args.rirs_dir.exists():
        LOGGER.warning("RIRS directory missing; running without reverb: %s", args.rirs_dir)

    if not (0.0 <= args.reverb_probability <= 1.0):
        raise RuntimeError("--reverb-probability must be in [0.0, 1.0]")
    if not args.snrs:
        raise RuntimeError("At least one SNR value is required.")

    source_files = _list_audio_files(args.input_dir)
    if not source_files:
        raise RuntimeError(f"No audio files found in input directory: {args.input_dir}")

    noise_files = _list_audio_files(args.musan_dir)
    if not noise_files:
        raise RuntimeError(f"No MUSAN noise files found: {args.musan_dir}")

    rir_files = _list_audio_files(args.rirs_dir) if args.rirs_dir.exists() else []
    rng = random.Random(args.seed)

    LOGGER.info("Source files: %d", len(source_files))
    LOGGER.info("MUSAN noise files: %d", len(noise_files))
    LOGGER.info("RIRS files: %d", len(rir_files))
    LOGGER.info("SNR choices (dB): %s", ", ".join(str(v) for v in args.snrs))

    from tqdm import tqdm

    for src_path in tqdm(source_files, desc="augment_audio"):
        rel = src_path.relative_to(args.input_dir)
        stem = _safe_name(rel.stem)
        out_rel = rel.with_name(f"{stem}_aug.wav")
        out_path = args.output_dir / out_rel
        _augment_one(
            src_audio=src_path,
            noise_files=noise_files,
            rir_files=rir_files,
            out_path=out_path,
            rng=rng,
            snr_choices=args.snrs,
            reverb_probability=args.reverb_probability,
            force=args.force,
        )

    LOGGER.info("Augmentation completed. Output directory: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1)
