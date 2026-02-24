#!/usr/bin/env python3
"""
OpenSLR downloader with resumable HTTP download and archive extraction.

Hardcoded targets:
- https://www.openslr.org/resources/12/dev-clean.tar.gz
- https://www.openslr.org/resources/12/dev-other.tar.gz
- https://www.openslr.org/resources/17/musan.tar.gz
- https://www.openslr.org/resources/28/rirs_noises.zip
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple


LOGGER = logging.getLogger("openslr_download")

OPENSLR_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "musan": "https://www.openslr.org/resources/17/musan.tar.gz",
    "rirs_noises": "https://www.openslr.org/resources/28/rirs_noises.zip",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_filename(url: str, expected_filename: Optional[str]) -> str:
    if expected_filename:
        return expected_filename
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise RuntimeError(f"Could not infer filename from URL: {url}")
    return name


def _head(url: str) -> tuple[Optional[int], bool]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        content_length_raw = response.headers.get("Content-Length")
        accept_ranges = (response.headers.get("Accept-Ranges") or "").lower()
        content_length = int(content_length_raw) if content_length_raw and content_length_raw.isdigit() else None
        supports_range = "bytes" in accept_ranges
        return content_length, supports_range


def _stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing_bytes = destination.stat().st_size if destination.exists() else 0
    content_length = None
    supports_range = False
    try:
        content_length, supports_range = _head(url)
    except Exception:
        # Some servers reject HEAD. We still try a GET.
        LOGGER.warning("HEAD request failed for %s; continuing with GET.", url)

    if content_length is not None and existing_bytes == content_length and content_length > 0:
        LOGGER.info("Download already complete: %s", destination)
        return

    range_start = existing_bytes if (existing_bytes > 0 and supports_range) else 0
    request = urllib.request.Request(url)
    if range_start > 0:
        request.add_header("Range", f"bytes={range_start}-")
        LOGGER.info("Resuming download at byte %d: %s", range_start, destination.name)
    else:
        LOGGER.info("Downloading: %s", destination.name)

    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", 200)
        append_mode = range_start > 0 and status == 206
        mode = "ab" if append_mode else "wb"
        if range_start > 0 and not append_mode:
            LOGGER.info("Server ignored Range; restarting download from 0.")

        with destination.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

    final_size = destination.stat().st_size if destination.exists() else 0
    if final_size <= 0:
        raise RuntimeError(f"Download failed or empty file: {destination}")
    if content_length is not None and final_size != content_length:
        raise RuntimeError(
            f"Downloaded size mismatch for {destination.name}: expected {content_length}, got {final_size}"
        )


def _hash_file(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest().lower()


def _parse_checksum_line(line: str, filename: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped:
        return None

    # Format: <hash>  <filename> OR <hash> *<filename>
    match = re.match(r"^([A-Fa-f0-9]{32,128})\s+\*?(.+)$", stripped)
    if match:
        digest = match.group(1).lower()
        referenced_name = Path(match.group(2).strip()).name
        if referenced_name == filename:
            algorithm = "md5" if len(digest) == 32 else "sha256" if len(digest) == 64 else None
            if algorithm:
                return algorithm, digest

    # Format: MD5 (filename) = <hash>
    match = re.match(r"^(MD5|SHA256)\s+\(([^)]+)\)\s*=\s*([A-Fa-f0-9]{32,128})$", stripped, re.IGNORECASE)
    if match:
        referenced_name = Path(match.group(2).strip()).name
        if referenced_name == filename:
            algo = match.group(1).lower()
            digest = match.group(3).lower()
            algorithm = "md5" if algo == "md5" else "sha256"
            return algorithm, digest

    # Format: only the hash (common in single-file *.md5/*.sha256)
    match = re.match(r"^([A-Fa-f0-9]{32}|[A-Fa-f0-9]{64})$", stripped)
    if match:
        digest = match.group(1).lower()
        algorithm = "md5" if len(digest) == 32 else "sha256"
        return algorithm, digest

    return None


def _maybe_get_checksum(url: str, filename: str) -> Optional[Tuple[str, str, str]]:
    base = url.rsplit("/", 1)[0]
    checksum_candidates = [
        (f"{url}.sha256", "sha256"),
        (f"{url}.md5", "md5"),
        (f"{base}/sha256sum.txt", None),
        (f"{base}/md5sum.txt", None),
        (f"{base}/checksums.txt", None),
    ]

    for checksum_url, forced_algo in checksum_candidates:
        try:
            with urllib.request.urlopen(checksum_url) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

        for line in body.splitlines():
            parsed = _parse_checksum_line(line, filename)
            if not parsed:
                continue
            algorithm, expected_hash = parsed
            if forced_algo and algorithm != forced_algo:
                continue
            return algorithm, expected_hash, checksum_url

    return None


def _verify_checksum_if_available(url: str, file_path: Path, filename: str) -> None:
    checksum = _maybe_get_checksum(url, filename)
    if not checksum:
        size = file_path.stat().st_size if file_path.exists() else 0
        if size <= 0:
            raise RuntimeError(f"Downloaded file is empty: {file_path}")
        LOGGER.info("No checksum file discovered; validated non-empty archive: %s", file_path.name)
        return

    algorithm, expected_hash, checksum_url = checksum
    actual_hash = _hash_file(file_path, algorithm)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{algorithm.upper()} mismatch for {file_path.name}.\n"
            f"Expected: {expected_hash}\nActual:   {actual_hash}\nSource:   {checksum_url}"
        )
    LOGGER.info("%s verified for %s (%s)", algorithm.upper(), file_path.name, checksum_url)


def _extract_archive(archive_path: Path, out_dir: Path) -> None:
    archive_name = archive_path.name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    if archive_name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, mode="r:gz") as tf:
            members = tf.getmembers()
            extracted_count = len(members)
            tf.extractall(out_dir)
    elif archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            members = zf.infolist()
            extracted_count = len(members)
            zf.extractall(out_dir)
    else:
        raise RuntimeError(
            f"Unsupported archive format: {archive_path}. Expected .tar.gz/.tgz or .zip"
        )

    if extracted_count <= 0:
        raise RuntimeError(f"Archive extraction produced no entries: {archive_path}")


def download_and_extract(url: str, out_dir: Path, expected_filename: Optional[str] = None) -> Path:
    """
    Download (resumable when Range is supported), verify checksum when available,
    then extract archive content into out_dir.
    """
    out_dir = Path(out_dir)
    filename = _resolve_filename(url, expected_filename)
    archive_path = out_dir / filename

    _stream_download(url, archive_path)
    _verify_checksum_if_available(url, archive_path, filename)
    _extract_archive(archive_path, out_dir)

    LOGGER.info("Download + extraction complete: %s", filename)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract OpenSLR archives.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "openslr",
        help="Base output directory for OpenSLR downloads.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "dev-clean", "dev-other", "musan", "rirs_noises"],
        default="all",
        help="Which dataset to download.",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    targets = (
        OPENSLR_URLS.items()
        if args.dataset == "all"
        else [(args.dataset, OPENSLR_URLS[args.dataset])]
    )

    target_dirs = {
        "dev-clean": args.out_dir / "SLR12_LibriSpeech",
        "dev-other": args.out_dir / "SLR12_LibriSpeech",
        "musan": args.out_dir / "SLR17_MUSAN",
        "rirs_noises": args.out_dir / "SLR28_RIRS_NOISES",
    }

    for name, url in targets:
        out_dir = target_dirs[name]
        out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Processing %s -> %s", name, out_dir)
        download_and_extract(url=url, out_dir=out_dir)

    LOGGER.info("All requested OpenSLR downloads completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1)
